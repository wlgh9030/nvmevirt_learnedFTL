// SPDX-License-Identifier: GPL-2.0-only

#include <linux/vmalloc.h>
#include <linux/ktime.h>
#include <linux/sched/clock.h>
#include <linux/math64.h>

#include "nvmev.h"
#include "conv_ftl.h"

static atomic64_t cmt_hits = ATOMIC64_INIT(0);
static atomic64_t cmt_misses = ATOMIC64_INIT(0);
static atomic64_t tp_loads = ATOMIC64_INIT(0);
static atomic64_t tp_writebacks = ATOMIC64_INIT(0);
static atomic64_t cmt_node_evicts = ATOMIC64_INIT(0);
static atomic64_t tp_writeback_entries_total = ATOMIC64_INIT(0);

static atomic64_t model_trains = ATOMIC64_INIT(0);
static atomic64_t model_bits_set = ATOMIC64_INIT(0);
static atomic64_t model_uses = ATOMIC64_INIT(0);
static atomic64_t model_hits = ATOMIC64_INIT(0);
static atomic64_t si_installs = ATOMIC64_INIT(0); /* LearnedFTL sequential-init model installs */
/* GC/training diagnostics */
static atomic64_t gc_runs = ATOMIC64_INIT(0);
static atomic64_t gc_reloc_pages = ATOMIC64_INIT(0);
static atomic64_t gc_groups = ATOMIC64_INIT(0);
static atomic64_t gc_group_targeted =
	ATOMIC64_INIT(0); /* do_gc passes that targeted a real group */
static atomic64_t gc_fallback = ATOMIC64_INIT(0); /* do_gc passes that fell back to global victim */
static atomic64_t host_writes = ATOMIC64_INIT(
	0); /* host data page writes; WA = (host_writes + gc_reloc_pages) / host_writes */

/* Runtime read-path counters for /proc/nvmev/cmt_stat. access = CMT hit + miss.
 * Reading reports CMT stats; writing resets the per-phase CMT and LR model use/hit
 * counters so one benchmark phase can be measured in isolation. These touch only
 * diagnostic counters, so they never affect mapping/GC behavior. */
void conv_cmt_stat_read(uint64_t *access, uint64_t *hit, uint64_t *miss)
{
	uint64_t h = atomic64_read(&cmt_hits);
	uint64_t m = atomic64_read(&cmt_misses);

	*hit = h;
	*miss = m;
	*access = h + m;
}

void conv_cmt_stat_reset(void)
{
	atomic64_set(&cmt_hits, 0);
	atomic64_set(&cmt_misses, 0);
	atomic64_set(&model_uses, 0);
	atomic64_set(&model_hits, 0);
}

static inline bool last_pg_in_wordline(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	return (ppa->g.pg % spp->pgs_per_oneshotpg) == (spp->pgs_per_oneshotpg - 1);
}

static bool should_gc(struct conv_ftl *conv_ftl)
{
	return (conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines);
}

static inline bool should_gc_high(struct conv_ftl *conv_ftl)
{
	return conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines_high;
}
/*
static inline struct ppa get_maptbl_ent(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return conv_ftl->maptbl[lpn];
}

static inline void set_maptbl_ent(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *ppa)
{
	NVMEV_ASSERT(lpn < conv_ftl->ssd->sp.tt_pgs);
	conv_ftl->maptbl[lpn] = *ppa;
}
*/
static uint64_t ppa2pgidx(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint64_t oneshot = spp->pgs_per_oneshotpg;
	uint64_t pu = (uint64_t)spp->nchs *
		      spp->luns_per_ch; /* parallel blocks per line (pls_per_lun==1) */
	uint64_t wl = ppa->g.pg / oneshot; /* wordline within block */
	uint64_t pg_off = ppa->g.pg % oneshot; /* page within a oneshot (one channel run) */
	uint64_t pgidx;

	NVMEV_DEBUG_VERBOSE("%s: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d\n", __func__, ppa->g.ch,
			    ppa->g.lun, ppa->g.pl, ppa->g.blk, ppa->g.pg);

	/*
	 * Allocation-order "virtual PPN": lay out (ch,lun,blk,pg) so that
	 * advance_write_pointer's write order (pg_off -> ch -> lun -> wordline -> blk)
	 * produces *consecutive* indices. A sequential write stream then yields a
	 * sequential pgidx, so the learned index sees a (near-)linear lpn->pgidx instead
	 * of the channel-striped sawtooth the old ch-major layout produced. Physical
	 * striping (ch/lun) sits in the low digits; the per-channel page progression
	 * (wordline, blk) in the high digits. pgidx2ppa() is the exact inverse.
	 * Assumes pls_per_lun == 1 (matches advance_write_pointer / the TODO there).
	 */
	pgidx = (uint64_t)ppa->g.blk * spp->pgs_per_line + wl * (oneshot * pu) +
		(uint64_t)ppa->g.lun * (oneshot * spp->nchs) + (uint64_t)ppa->g.ch * oneshot +
		pg_off;

	NVMEV_ASSERT(pgidx < spp->tt_pgs);

	return pgidx;
}

/* inverse of ppa2pgidx: reconstruct a ppa from the allocation-order virtual PPN */
static struct ppa pgidx2ppa(struct conv_ftl *conv_ftl, uint64_t pgidx)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint64_t oneshot = spp->pgs_per_oneshotpg;
	uint64_t pu = (uint64_t)spp->nchs * spp->luns_per_ch;
	uint64_t wl, pg_off;
	struct ppa ppa;

	// ppa.ppa = 0;
	// ppa.g.ch = pgidx / spp->pgs_per_ch;
	// pgidx %= spp->pgs_per_ch;
	// ppa.g.lun = pgidx / spp->pgs_per_lun;
	// pgidx %= spp->pgs_per_lun;
	// ppa.g.pl = pgidx / spp->pgs_per_pl;
	// pgidx %= spp->pgs_per_pl;
	// ppa.g.blk = pgidx / spp->pgs_per_blk;
	// pgidx %= spp->pgs_per_blk;
	// ppa.g.pg = pgidx;

	ppa.ppa = 0;
	ppa.g.blk = pgidx / spp->pgs_per_line;
	pgidx %= spp->pgs_per_line;
	wl = pgidx / (oneshot * pu);
	pgidx %= (oneshot * pu);
	ppa.g.lun = pgidx / (oneshot * spp->nchs);
	pgidx %= (oneshot * spp->nchs);
	ppa.g.ch = pgidx / oneshot;
	pg_off = pgidx % oneshot;
	ppa.g.pl = 0;
	ppa.g.pg = wl * oneshot + pg_off;

	return ppa;
}

static inline uint64_t get_rmap_ent(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	uint64_t pgidx = ppa2pgidx(conv_ftl, ppa);

	return conv_ftl->rmap[pgidx];
}

/* set rmap[page_no(ppa)] -> lpn */
static inline void set_rmap_ent(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *ppa)
{
	uint64_t pgidx = ppa2pgidx(conv_ftl, ppa);

	conv_ftl->rmap[pgidx] = lpn;
}

static inline int victim_line_cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
	return (next > curr);
}

static inline pqueue_pri_t victim_line_get_pri(void *a)
{
	return ((struct line *)a)->vpc;
}

static inline void victim_line_set_pri(void *a, pqueue_pri_t pri)
{
	((struct line *)a)->vpc = pri;
}

static inline size_t victim_line_get_pos(void *a)
{
	return ((struct line *)a)->pos;
}

static inline void victim_line_set_pos(void *a, size_t pos)
{
	((struct line *)a)->pos = pos;
}

static inline void consume_write_credit(struct conv_ftl *conv_ftl)
{
	conv_ftl->wfc.write_credits--;
}

static void foreground_gc(struct conv_ftl *conv_ftl);
static int do_gc(struct conv_ftl *conv_ftl, bool force);

static inline void check_and_refill_write_credit(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	if (wfc->write_credits <= 0) {
		foreground_gc(conv_ftl);

		wfc->write_credits += wfc->credits_to_refill;
	}
}

static void init_lines(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *line;
	int i;

	lm->tt_lines = spp->blks_per_pl;
	NVMEV_ASSERT(lm->tt_lines == spp->tt_lines);
	lm->lines = vmalloc(sizeof(struct line) * lm->tt_lines);

	INIT_LIST_HEAD(&lm->free_line_list);
	INIT_LIST_HEAD(&lm->full_line_list);

	lm->victim_line_pq = pqueue_init(spp->tt_lines, victim_line_cmp_pri, victim_line_get_pri,
					 victim_line_set_pri, victim_line_get_pos,
					 victim_line_set_pos);

	lm->free_line_cnt = 0;
	for (i = 0; i < lm->tt_lines; i++) {
		lm->lines[i] = (struct line){
			.id = i,
			.ipc = 0,
			.vpc = 0,
			.group = GROUP_NONE,
			.pos = 0,
			.entry = LIST_HEAD_INIT(lm->lines[i].entry),
		};

		/* initialize all the lines as free lines */
		list_add_tail(&lm->lines[i].entry, &lm->free_line_list);
		lm->free_line_cnt++;
	}

	NVMEV_ASSERT(lm->free_line_cnt == lm->tt_lines);
	lm->victim_line_cnt = 0;
	lm->full_line_cnt = 0;
}

static void remove_lines(struct conv_ftl *conv_ftl)
{
	pqueue_free(conv_ftl->lm.victim_line_pq);
	vfree(conv_ftl->lm.lines);
}

static void init_write_flow_control(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	wfc->write_credits = spp->pgs_per_line;
	wfc->credits_to_refill = spp->pgs_per_line;
}

static inline void check_addr(int a, int max)
{
	NVMEV_ASSERT(a >= 0 && a < max);
}

static struct line *get_next_free_line(struct conv_ftl *conv_ftl)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *curline = list_first_entry_or_null(&lm->free_line_list, struct line, entry);

	if (!curline) {
		NVMEV_ERROR("No free line left in VIRT !!!!\n");
		return NULL;
	}

	list_del_init(&curline->entry);
	lm->free_line_cnt--;
	NVMEV_DEBUG("%s: free_line_cnt %d\n", __func__, lm->free_line_cnt);
	return curline;
}

/* USER_IO writes go through conv_ftl->group_wp[group_of(lpn)] directly; this
 * resolves the single GC / translation frontiers only. */
static struct write_pointer *__get_wp(struct conv_ftl *ftl, uint32_t io_type)
{
	if (io_type == GC_IO) {
		return &ftl->gc_wp;
	} else if (io_type == TRANS_IO) {
		return &ftl->trans_wp;
	}

	NVMEV_ASSERT(0);
	return NULL;
}

/* Lazy allocation: give wp a fresh line only when it has none (curline == NULL); wp->group
 * is set once at init and the grabbed line is tagged with it. Every get_new_page() caller
 * must call this first. Lazy (vs eager-at-init) is required under group==line geometry where
 * num_groups == num_lines: an idle or full group then pins no open line. */
static void ensure_write_pointer(struct conv_ftl *conv_ftl, struct write_pointer *wp, bool gc_flag)
{
	struct line *curline;

	if (wp->curline)
		return;

	/* ld-tpftl init_line_write_pointer: grab a fresh line only AFTER reclaiming when the
	 * free list is low. gc_flag=true (user writes) reclaims first so allocation never hits
	 * an empty free list under many open per-group frontiers; gc_flag=false (GC relocation,
	 * trans writeback) must NOT re-enter do_gc. do_gc reclaims a GC_BATCH_LINES batch, so a
	 * single call jumps free_line_cnt back above the threshold. */
	if (gc_flag && conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines)
		do_gc(conv_ftl, true);

	curline = get_next_free_line(conv_ftl);
	NVMEV_ASSERT(curline);
	curline->group = wp->group;
	/* per-group owned-line accounting (LearnedFTL cumulative_allocated_blocks): only data
	 * groups are counted; gc/trans frontiers are GROUP_NONE and don't drive group GC. */
	if (wp->group != GROUP_NONE)
		conv_ftl->groups[wp->group].alloc_lines++;
	wp->curline = curline;
	wp->ch = 0;
	wp->lun = 0;
	wp->pg = 0;
	wp->blk = curline->id;
	wp->pl = 0;
}

static void advance_write_pointer(struct conv_ftl *conv_ftl, struct write_pointer *wpp)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;

	NVMEV_DEBUG_VERBOSE("current wpp: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d\n", wpp->ch, wpp->lun,
			    wpp->pl, wpp->blk, wpp->pg);

	check_addr(wpp->pg, spp->pgs_per_blk);
	wpp->pg++;
	if ((wpp->pg % spp->pgs_per_oneshotpg) != 0)
		goto out;

	wpp->pg -= spp->pgs_per_oneshotpg;
	check_addr(wpp->ch, spp->nchs);
	wpp->ch++;
	if (wpp->ch != spp->nchs)
		goto out;

	wpp->ch = 0;
	check_addr(wpp->lun, spp->luns_per_ch);
	wpp->lun++;
	/* in this case, we should go to next lun */
	if (wpp->lun != spp->luns_per_ch)
		goto out;

	wpp->lun = 0;
	/* go to next wordline in the block */
	wpp->pg += spp->pgs_per_oneshotpg;
	if (wpp->pg != spp->pgs_per_blk)
		goto out;

	wpp->pg = 0;
	/* move current line to {victim,full} line list */
	if (wpp->curline->vpc == spp->pgs_per_line) {
		/* all pgs are still valid, move to full line list */
		NVMEV_ASSERT(wpp->curline->ipc == 0);
		list_add_tail(&wpp->curline->entry, &lm->full_line_list);
		lm->full_line_cnt++;
		NVMEV_DEBUG_VERBOSE("wpp: move line to full_line_list\n");
	} else {
		NVMEV_DEBUG_VERBOSE("wpp: line is moved to victim list\n");
		NVMEV_ASSERT(wpp->curline->vpc >= 0 && wpp->curline->vpc < spp->pgs_per_line);
		/* there must be some invalid pages in this line */
		NVMEV_ASSERT(wpp->curline->ipc > 0);
		pqueue_insert(lm->victim_line_pq, wpp->curline);
		lm->victim_line_cnt++;
	}
	/* lazy frontier: the full line was sealed above; don't grab a new one here. The next
	 * ensure_write_pointer() grabs one when this frontier is next written, so an idle or
	 * full group holds no open line — required when num_groups == num_lines (group==line). */
	wpp->curline = NULL;
	return;
out:
	NVMEV_DEBUG_VERBOSE("advanced wpp: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d (curline %d)\n",
			    wpp->ch, wpp->lun, wpp->pl, wpp->blk, wpp->pg, wpp->curline->id);
}

static struct ppa get_new_page(struct conv_ftl *conv_ftl, struct write_pointer *wp)
{
	struct ppa ppa;

	ppa.ppa = 0;
	ppa.g.ch = wp->ch;
	ppa.g.lun = wp->lun;
	ppa.g.pg = wp->pg;
	ppa.g.blk = wp->blk;
	ppa.g.pl = wp->pl;

	NVMEV_ASSERT(ppa.g.pl == 0);

	return ppa;
}
/*
static void init_maptbl(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->maptbl = vmalloc(sizeof(struct ppa) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		conv_ftl->maptbl[i].ppa = UNMAPPED_PPA;
	}
}

static void remove_maptbl(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->maptbl);
}
*/

static void mark_page_valid(struct conv_ftl *conv_ftl, struct ppa *ppa);
static void mark_page_invalid(struct conv_ftl *conv_ftl, struct ppa *ppa);
static inline bool mapped_ppa(struct ppa *ppa);
static struct tp_node *node_lookup(struct dftl_cmt *cmt, uint32_t tp_idx);
static void tp_flush_idx(struct conv_ftl *conv_ftl, uint32_t tp_idx, struct tp_node *node,
			 uint64_t *stime);
static struct line *get_line(struct conv_ftl *conv_ftl, struct ppa *ppa);

static inline uint32_t entries_per_tp(struct conv_ftl *conv_ftl)
{
	return conv_ftl->ssd->sp.pgsz / sizeof(struct ppa); // => 512
}

/* group id that owns lpn's translation page (LearnedFTL GTD-entry grouping) */
static inline uint32_t group_of(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return (uint32_t)((lpn / entries_per_tp(conv_ftl)) / TP_PER_GROUP);
}

/* ---- learned index: fixed-point linear regression (kernel has no FPU) ---- */

/* in-place sort of (x[], y[]) pairs by x ascending (quicksort, port of util.c) */
static void lr_sort_pairs(uint64_t *x, uint64_t *y, int low, int high)
{
	int i = low, j = high;
	uint64_t px = x[(low + high) / 2];

	while (i <= j) {
		while (x[i] < px)
			i++;
		while (x[j] > px)
			j--;
		if (i <= j) {
			uint64_t tx = x[i], ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
			i++;
			j--;
		}
	}
	if (low < j)
		lr_sort_pairs(x, y, low, j);
	if (i < high)
		lr_sort_pairs(x, y, i, high);
}

/*
 * Least squares fit of y = w*x + b over num points, returning w,b as
 * LR_FP_SHIFT fixed-point. Returns false if the system is degenerate.
 * x values are small (0..ents_per_tp) and y are group-normalized, so the
 * intermediate sums fit comfortably in 64-bit signed integers.
 */
static bool lr_least_square(const uint64_t *x, const uint64_t *y, int num, int64_t *w_fp,
			    int64_t *b_fp)
{
	int64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0; /* sum x^2, sum x, sum xy, sum y */
	int64_t den;
	int i;

	if (num <= 0)
		return false;

	for (i = 0; i < num; i++) {
		int64_t xi = (int64_t)x[i];
		int64_t yi = (int64_t)y[i];

		t1 += xi * xi;
		t2 += xi;
		t3 += xi * yi;
		t4 += yi;
	}

	den = t1 * num - t2 * t2;
	if (den == 0)
		return false;

	*w_fp = div64_s64((t3 * num - t2 * t4) << LR_FP_SHIFT, den);
	*b_fp = div64_s64((t1 * t4 - t2 * t3) << LR_FP_SHIFT, den);
	return true;
}

/* evaluate the fixed-point model at x with round-to-nearest, clamped at 0 */
static uint64_t lr_predict(uint64_t x, int64_t w_fp, int64_t b_fp)
{
	int64_t y = w_fp * (int64_t)x + b_fp;

	y += (1LL << (LR_FP_SHIFT - 1)); /* round to nearest */
	y >>= LR_FP_SHIFT;
	if (y < 0)
		y = 0;
	return (uint64_t)y;
}

/*
 * Train the model for one translation page from n (lpn, pgidx) samples that all
 * belong to the same tp_idx. lpns/pgidxs are sorted by lpn ascending and are
 * normalized in place. Port of FEMU model_training() single-group path:
 * split into LR_MAX_INTERVALS piecewise segments, least-squares fit each, and
 * mark bitmaps[lpn]=1 for samples the model reproduces exactly.
 */
static void lr_train_tp(struct conv_ftl *conv_ftl, uint64_t *lpns, uint64_t *pgidxs, int n)
{
	uint32_t ents_per_tp = entries_per_tp(conv_ftl);
	uint32_t tp_idx = lpns[0] / ents_per_tp;
	struct lr_node *node = &conv_ftl->lr_nodes[tp_idx];
	uint64_t start_lpn = lpns[0];
	uint64_t start_ppa = pgidxs[0];
	int interval_num, j, i;
	uint32_t total_su = 0;

	if (n <= LR_TRAIN_THRESHOLD)
		return;

	/* keep-longer (LearnedFTL SI step ④, applied to GC training too): a GC fit can
	 * predict at most n LPNs exactly, so if the current model already covers >= n
	 * (e.g. a sequential-init y=x), don't let this smaller batch replace it. */
	if (n <= node->cover_len)
		return;

	interval_num = n / LR_MAX_INTERVALS;
	if (interval_num == 0)
		return;

	/* normalize samples relative to the segment origin */
	for (i = 0; i < n; i++) {
		lpns[i] -= start_lpn;
		pgidxs[i] -= start_ppa;
	}

	node->start_lpn = start_lpn;
	node->start_ppa = start_ppa;

	for (j = 0; j < LR_MAX_INTERVALS; j++) {
		int start = (j == 0) ? 0 : (j * interval_num);
		int end = (j == LR_MAX_INTERVALS - 1) ? n : ((j + 1) * interval_num);
		int num_p = end - start;
		int64_t w_fp = 0, b_fp = 0;
		uint32_t su = 0;

		if (num_p <= 0) {
			node->brks[j].key = (j == 0) ? 0 : node->brks[j - 1].key;
			node->brks[j].w_fp = 0;
			node->brks[j].b_fp = 0;
			node->brks[j].valid_cnt = 0;
			continue;
		}

		/* segment covers normalized lpns up to (inclusive) the last sample */
		node->brks[j].key = lpns[end - 1];

		if (lr_least_square(&lpns[start], &pgidxs[start], num_p, &w_fp, &b_fp)) {
			node->brks[j].w_fp = w_fp;
			node->brks[j].b_fp = b_fp;
			for (i = start; i < end; i++) {
				uint64_t pred = lr_predict(lpns[i], w_fp, b_fp);
				if (pred == pgidxs[i]) {
					su++;
					conv_ftl->bitmaps[start_lpn + lpns[i]] = 1;
				} else {
					conv_ftl->bitmaps[start_lpn + lpns[i]] = 0;
				}
			}
		} else {
			node->brks[j].w_fp = 0;
			node->brks[j].b_fp = 0;
		}
		node->brks[j].valid_cnt = su;
		total_su += su;
		atomic64_add(su, &model_bits_set);
	}

	node->cover_len = total_su;
	node->u = 1;
	atomic64_inc(&model_trains);
}

/* train models from samples collected during the just-finished line GC */
static void lr_train_collected(struct conv_ftl *conv_ftl)
{
	uint32_t ents_per_tp = entries_per_tp(conv_ftl);
	int total = conv_ftl->gc_train_cnt;
	int i, gstart;

	atomic64_add(total, &gc_reloc_pages);
	if (total <= 0)
		return;

	/* sort by lpn so that same-tp samples are contiguous */
	lr_sort_pairs(conv_ftl->gc_train_lpns, conv_ftl->gc_train_pgidxs, 0, total - 1);

	gstart = 0;
	for (i = 1; i <= total; i++) {
		bool boundary = (i == total) || (conv_ftl->gc_train_lpns[i] / ents_per_tp !=
						 conv_ftl->gc_train_lpns[gstart] / ents_per_tp);
		if (boundary) {
			atomic64_inc(&gc_groups);
			/* LearnedFTL처럼 데이터 GC에서는 모델만 재학습하고 translation
			 * page는 NAND에 내리지 않는다. 매핑 정답은 tp_map(write-through)에
			 * 이미 반영돼 있고, NAND 영속화는 trans frontier의 writeback이 담당. */
			lr_train_tp(conv_ftl, &conv_ftl->gc_train_lpns[gstart],
				    &conv_ftl->gc_train_pgidxs[gstart], i - gstart);
			gstart = i;
		}
	}

	conv_ftl->gc_train_cnt = 0;
}

static void init_dftl(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct dftl_cmt *cmt = &conv_ftl->cmt;
	int i;

	conv_ftl->num_tp = DIV_ROUND_UP(spp->tt_pgs, entries_per_tp(conv_ftl));

	/* GTD-entry groups for group-granular GC (LearnedFTL) */
	conv_ftl->num_groups = DIV_ROUND_UP(conv_ftl->num_tp, TP_PER_GROUP);
	conv_ftl->groups = vzalloc(sizeof(struct gtd_group) * conv_ftl->num_groups);
	conv_ftl->group_wp = vmalloc(sizeof(struct write_pointer) * conv_ftl->num_groups);

	/* free-line reserve (lazy-frontier / group==line model): GC keeps ~this many lines
	 * free. NOT num_groups (== tt_lines here, which would exceed the device). With lazy
	 * frontiers an idle/full group pins no open line, so a modest reserve suffices; raise
	 * it or OP_AREA_PERCENT if random-write open-frontier spikes drain the free list. */
	/* MUST be well below (tt_lines - namespace_lines) or GC triggers every write: with
	 * tt_lines=512, namespace~393, free maxes at ~119, so a 128 reserve made free<=reserve
	 * ALWAYS true -> constant GC -> WA~2.8 + channel flood -> timeout. 32 leaves ample
	 * working room while only firing when the free list is genuinely low. */
	conv_ftl->cp.gc_thres_lines = 32;
	conv_ftl->cp.gc_thres_lines_high = 32;

	NVMEV_INFO("FTL geometry: tt_pgs=%ld tt_lines=%d pgs_per_line=%d ents_per_tp=%u "
		   "num_tp=%u TP_PER_GROUP=%d num_groups=%u frontiers=%u reserve=%u\n",
		   (long)spp->tt_pgs, (int)spp->tt_lines, (int)spp->pgs_per_line,
		   entries_per_tp(conv_ftl), conv_ftl->num_tp, TP_PER_GROUP, conv_ftl->num_groups,
		   conv_ftl->num_groups + 2, conv_ftl->cp.gc_thres_lines);

	/* free-line safety margin: host-writable lines + reserve must fit in tt_lines,
	 * with enough slack to absorb partial-fill waste of the open frontiers */
	{
		uint32_t ns_lines = (uint32_t)((uint64_t)spp->tt_pgs * 100 /
					       conv_ftl->cp.pba_pcent / spp->pgs_per_line);
		int usable = (int)spp->tt_lines - (int)conv_ftl->cp.gc_thres_lines;

		NVMEV_INFO(
			"FTL line budget: namespace~%u lines, usable=%d (tt_lines %d - reserve %u), "
			"slack=%d, frontiers=%u\n",
			ns_lines, usable, (int)spp->tt_lines, conv_ftl->cp.gc_thres_lines,
			usable - (int)ns_lines, conv_ftl->num_groups + 2);
		if (usable - (int)ns_lines < (int)conv_ftl->num_groups)
			NVMEV_INFO("FTL WARNING: line slack < num_groups — raise OP_AREA_PERCENT "
				   "if GC wedges under full-device writes\n");
	}

	conv_ftl->gtd = vmalloc(sizeof(struct ppa) * conv_ftl->num_tp);
	for (i = 0; i < conv_ftl->num_tp; i++) {
		conv_ftl->gtd[i].ppa = UNMAPPED_PPA;
	}

	/* TP content store: one page per TP, indexed by tp_idx, init UNMAPPED */
	conv_ftl->tp_map = vmalloc((size_t)conv_ftl->num_tp * spp->pgsz);
	{
		struct ppa *p = (struct ppa *)conv_ftl->tp_map;
		uint64_t n = (uint64_t)conv_ftl->num_tp * entries_per_tp(conv_ftl);
		uint64_t k;
		for (k = 0; k < n; k++)
			p[k].ppa = UNMAPPED_PPA;
	}

	/* learned-index models: one per TP, all untrained (u=0) initially */
	conv_ftl->lr_nodes = vmalloc(sizeof(struct lr_node) * conv_ftl->num_tp);
	for (i = 0; i < conv_ftl->num_tp; i++) {
		conv_ftl->lr_nodes[i].u = 0;
		conv_ftl->lr_nodes[i].cover_len = 0;
	}

	/* sequential-initialization run state starts empty */
	conv_ftl->si_run_len = 0;

	/* per-lpn prediction bitmap, init all zero */
	conv_ftl->bitmaps = vzalloc(spp->tt_pgs);

	/* GC training sample buffers: a whole-group sweep collects all of one group's live
	 * data pages (<= TP_PER_GROUP * ents_per_tp) plus a small global top-up batch. */
	conv_ftl->gc_train_cap =
		TP_PER_GROUP * entries_per_tp(conv_ftl) + GC_BATCH_LINES * spp->pgs_per_line;
	conv_ftl->gc_train_lpns = vmalloc(sizeof(uint64_t) * (size_t)conv_ftl->gc_train_cap);
	conv_ftl->gc_train_pgidxs = vmalloc(sizeof(uint64_t) * (size_t)conv_ftl->gc_train_cap);
	conv_ftl->gc_train_cnt = 0;

	cmt->entry_pool = vmalloc(sizeof(struct cmt_entry) * CMT_CAPACITY);
	cmt->node_pool = vmalloc(sizeof(struct tp_node) * CMT_CAPACITY);
	cmt->entry_capacity = CMT_CAPACITY;
	cmt->node_capacity = CMT_CAPACITY;
	cmt->entry_size = 0;
	cmt->node_size = 0;
	hash_init(cmt->entry_ht);
	hash_init(cmt->node_ht);
	INIT_LIST_HEAD(&cmt->node_lru);
	INIT_LIST_HEAD(&cmt->entry_free_list);
	INIT_LIST_HEAD(&cmt->node_free_list);
	for (i = 0; i < CMT_CAPACITY; i++) {
		list_add(&cmt->entry_pool[i].sibling, &cmt->entry_free_list);
		list_add(&cmt->node_pool[i].lru, &cmt->node_free_list);
	}
}

static void remove_dftl(struct conv_ftl *conv_ftl)
{
	uint64_t wb = atomic64_read(&tp_writebacks);
	uint64_t wb_entries = atomic64_read(&tp_writeback_entries_total);

	NVMEV_INFO("CMT hits=%lld misses=%lld tp_loads=%lld tp_writebacks=%lld "
		   "node_evicts=%lld avg_batch=%lld.%02lld\n",
		   atomic64_read(&cmt_hits), atomic64_read(&cmt_misses), atomic64_read(&tp_loads),
		   wb, atomic64_read(&cmt_node_evicts), wb ? wb_entries / wb : 0,
		   wb ? (wb_entries * 100 / wb) % 100 : 0);

	NVMEV_INFO("LR model[%s]: trains=%lld bits_set=%lld uses=%lld hits=%lld si_installs=%lld\n",
		   LEARNED_INDEX_ENABLE ? "ON" : "OFF", atomic64_read(&model_trains),
		   atomic64_read(&model_bits_set), atomic64_read(&model_uses),
		   atomic64_read(&model_hits), atomic64_read(&si_installs));

	NVMEV_INFO("GC diag: runs=%lld reloc_pages=%lld groups=%lld under_thresh=%lld "
		   "group_gc=%lld fallback_gc=%lld\n",
		   atomic64_read(&gc_runs), atomic64_read(&gc_reloc_pages),
		   atomic64_read(&gc_groups),
		   atomic64_read(&gc_groups) - atomic64_read(&model_trains),
		   atomic64_read(&gc_group_targeted), atomic64_read(&gc_fallback));

	{
		uint64_t hw = atomic64_read(&host_writes);
		uint64_t rl = atomic64_read(&gc_reloc_pages);
		uint64_t wa100 = hw ? (hw + rl) * 100 / hw : 100;

		NVMEV_INFO("WA: host_writes=%llu gc_reloc_pages=%llu WA=%llu.%02llu\n",
			   (unsigned long long)hw, (unsigned long long)rl,
			   (unsigned long long)(wa100 / 100), (unsigned long long)(wa100 % 100));
	}

	vfree(conv_ftl->gtd);
	vfree(conv_ftl->tp_map);
	vfree(conv_ftl->lr_nodes);
	vfree(conv_ftl->bitmaps);
	vfree(conv_ftl->gc_train_lpns);
	vfree(conv_ftl->gc_train_pgidxs);
	vfree(conv_ftl->groups);
	vfree(conv_ftl->group_wp);
	vfree(conv_ftl->cmt.entry_pool);
	vfree(conv_ftl->cmt.node_pool);
}

/* Phase 1 verification: independently recount valid DATA pages per group by
 * scanning all physical pages, and compare against the maintained groups[]
 * counters. Valid pages keep rmap==lpn, so this is a fully independent check.
 * Must run before rmap/lines are torn down. */
static void dftl_group_stats_check(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t *scan;
	uint64_t pgidx;
	uint32_t g;
	bool ok = true;

	scan = vzalloc(sizeof(uint32_t) * conv_ftl->num_groups);
	if (!scan)
		return;

	for (pgidx = 0; pgidx < spp->tt_pgs; pgidx++) {
		struct ppa ppa = pgidx2ppa(conv_ftl, pgidx);
		struct nand_page *pg = get_pg(conv_ftl->ssd, &ppa);
		uint64_t lpn;

		if (pg->status != PG_VALID)
			continue;
		lpn = get_rmap_ent(conv_ftl, &ppa);
		if (lpn >= TRANS_LPN_BASE) /* translation page or unmapped */
			continue;
		{
			uint32_t gid = group_of(conv_ftl, lpn);
			struct line *ln = get_line(conv_ftl, &ppa);

			scan[gid]++;
			/* a valid DATA page must live either in its own group's line
			 * or in a GROUP_NONE line (GC-relocated via gc frontier) */
			if (ln->group != GROUP_NONE && ln->group != (int)gid) {
				NVMEV_INFO("GROUP ROUTING VIOLATION pgidx=%llu lpn=%llu "
					   "line.group=%d expected=%u\n",
					   (unsigned long long)pgidx, (unsigned long long)lpn,
					   ln->group, gid);
				ok = false;
			}
		}
	}

	for (g = 0; g < conv_ftl->num_groups; g++) {
		if (scan[g] != conv_ftl->groups[g].valid_pages) {
			NVMEV_INFO("GROUP STATS MISMATCH g=%u maintained=%u scan=%u\n", g,
				   conv_ftl->groups[g].valid_pages, scan[g]);
			ok = false;
		}
	}

	/* invalid_pages must equal the sum of ipc over each group's owned lines */
	memset(scan, 0, sizeof(uint32_t) * conv_ftl->num_groups);
	for (pgidx = 0; pgidx < conv_ftl->lm.tt_lines; pgidx++) {
		struct line *ln = &conv_ftl->lm.lines[pgidx];

		if (ln->group != GROUP_NONE)
			scan[ln->group] += (uint32_t)ln->ipc;
	}
	for (g = 0; g < conv_ftl->num_groups; g++) {
		if (scan[g] != conv_ftl->groups[g].invalid_pages) {
			NVMEV_INFO("GROUP INVALID MISMATCH g=%u maintained=%u scan=%u\n", g,
				   conv_ftl->groups[g].invalid_pages, scan[g]);
			ok = false;
		}
	}

	NVMEV_INFO("GROUP STATS %s (%u groups)\n", ok ? "OK" : "FAIL", conv_ftl->num_groups);
	vfree(scan);
}

static struct tp_node *node_lookup(struct dftl_cmt *cmt, uint32_t tp_idx)
{
	struct tp_node *node;

	hash_for_each_possible(cmt->node_ht, node, hnode, tp_idx) {
		if (node->tp_idx == tp_idx)
			return node;
	}
	return NULL;
}

/*
 * Flush translation page `tp_idx` to NAND. The page content is already current
 * in tp_map (write-through via dftl_set_ppa / dftl_set_ppa_gc), so no entry copy
 * is needed — this only models the read-modify-write + NAND write latency, moves
 * the TP to a new physical page, and updates the GTD. If the TP is cached
 * (`node` != NULL), the whole page is now persisted so its entries become clean.
 */
static void tp_flush_idx(struct conv_ftl *conv_ftl, uint32_t tp_idx, struct tp_node *node,
			 uint64_t *stime)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct ppa old_tp_ppa = conv_ftl->gtd[tp_idx];
	struct ppa new_tp_ppa;
	struct cmt_entry *e;
	struct nand_cmd cmd;
	uint64_t nsecs_completed, nsecs_latest = *stime;

	atomic64_inc(&tp_writebacks);
	if (node)
		atomic64_add(node->dirty_count, &tp_writeback_entries_total);

	/* 새 페이지 할당 — translation page 전용 frontier(trans_wp)에 쓴다.
	 * gc_flag=false: CMT eviction에 중첩될 수 있는 경로라 do_gc 재진입 차단 */
	ensure_write_pointer(conv_ftl, &conv_ftl->trans_wp, false);
	new_tp_ppa = get_new_page(conv_ftl, &conv_ftl->trans_wp);
	mark_page_valid(conv_ftl, &new_tp_ppa);
	advance_write_pointer(conv_ftl, &conv_ftl->trans_wp);

	if (mapped_ppa(&old_tp_ppa)) {
		/* 기존 TP read 지연 시뮬레이션 (read-modify-write) */
		cmd = (struct nand_cmd){
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = nsecs_latest,
			.xfer_size = spp->pgsz,
			.interleave_pci_dma = false,
			.ppa = &old_tp_ppa,
		};
		nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &cmd);
		nsecs_latest = max(nsecs_completed, nsecs_latest);

		/* 기존 페이지 무효화 */
		mark_page_invalid(conv_ftl, &old_tp_ppa);
		set_rmap_ent(conv_ftl, INVALID_LPN, &old_tp_ppa);
	}

	/* tp_map[tp_idx]는 write-through로 이미 최신 → 복사 불필요.
	 * 캐시된 노드가 있으면 이번 flush로 전부 영속화되므로 clean 처리. */
	if (node) {
		list_for_each_entry(e, &node->entries, sibling)
			e->dirty = false;
		node->dirty_count = 0;
	}

	/* rmap에 translation page임을 기록 */
	set_rmap_ent(conv_ftl, TRANS_LPN_BASE + tp_idx, &new_tp_ppa);

	/*
	 * NAND write 지연은 모델링하지 않는다 (원본 LearnedFTL translation_write_page와 동일).
	 * dirty 매핑 writeback은 이 요청의 critical path 밖(비동기로 숨겨짐)이라, program
	 * 지연을 트리거 요청에 동기로 부과하면 과대계상이 된다. 공간(free line) 소비는
	 * 아래 GC 정산으로 그대로 반영하므로 시간 모델만 생략한다.
	 * => 클로드 추측.. 아무튼 원본 LearnedFTL에서도 write latency 부분이 주석처리 되어 있음.
	 *	cmd = (struct nand_cmd){
	 *		.type = GC_IO, .cmd = NAND_NOP, .stime = nsecs_latest,
	 *		.interleave_pci_dma = false, .ppa = &new_tp_ppa,
	 *	};
	 *	if (last_pg_in_wordline(conv_ftl, &new_tp_ppa)) {
	 *		cmd.cmd = NAND_WRITE;
	 *		cmd.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
	 *	}
	 *	nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &cmd);
	 *	nsecs_latest = max(nsecs_completed, nsecs_latest);
	 */

	/* GTD 업데이트 */
	conv_ftl->gtd[tp_idx] = new_tp_ppa;

	*stime = nsecs_latest;

	/*
	 * TRANS_IO frontier도 물리 line을 소비하므로 user write(conv_write)와 동일하게
	 * write credit으로 정산한다. 이렇게 해야 모든 free-line 소비자(user write + TP
	 * writeback)가 같은 credit/GC로 reconcile되어 free line 고갈을 막는다.
	 * (원본의 pool-centric GC trigger와 동일한 효과.)
	 */
	consume_write_credit(conv_ftl);
	check_and_refill_write_credit(conv_ftl);
}

static void tp_writeback_node(struct conv_ftl *conv_ftl, struct tp_node *node, uint64_t *stime)
{
	tp_flush_idx(conv_ftl, node->tp_idx, node, stime);
}

static struct cmt_entry *cmt_lookup(struct dftl_cmt *cmt, uint64_t lpn)
{
	struct cmt_entry *entry;

	hash_for_each_possible(cmt->entry_ht, entry, hnode, lpn) { /* hash값에 해당하는 버킷에서 */
		if (entry->lpn == lpn) { /* 해당하는 lpn이 있으면 반환 */
			list_move(&entry->parent->lru, &cmt->node_lru); /* parent TPnode -> MRU */
			list_move(&entry->sibling, &entry->parent->entries);
			atomic64_inc(&cmt_hits);
			return entry;
		}
	}
	atomic64_inc(&cmt_misses);
	return NULL; /* 없으면 lookup failure */
}

/* LRU tail TPnode에서 entry 하나만 evict한다. dirty victim이면 TP 전체를 flush해
 * 같은 TPnode에 남은 dirty entries는 clean으로 유지한다. */
static void cmt_evict_entry(struct conv_ftl *conv_ftl, uint64_t *stime)
{
	struct dftl_cmt *cmt = &conv_ftl->cmt;
	struct tp_node *node;
	struct cmt_entry *e, *victim = NULL;

	node = list_last_entry(&cmt->node_lru, struct tp_node, lru);

	list_for_each_entry_reverse(e, &node->entries, sibling) {
		if (!e->dirty) {
			victim = e;
			break;
		}
	}

	if (!victim)
		victim = list_last_entry(&node->entries, struct cmt_entry, sibling);

	if (victim->dirty)
		tp_writeback_node(conv_ftl, node, stime);

	hash_del(&victim->hnode);
	list_del(&victim->sibling);
	victim->lpn = INVALID_LPN;
	victim->ppa.ppa = UNMAPPED_PPA;
	victim->dirty = false;
	victim->parent = NULL;
	list_add(&victim->sibling, &cmt->entry_free_list);
	node->entry_count--;
	cmt->entry_size--;

	if (node->entry_count == 0) {
		hash_del(&node->hnode);
		list_del(&node->lru);
		list_add(&node->lru, &cmt->node_free_list);
		cmt->node_size--;
		atomic64_inc(&cmt_node_evicts);
	}
}

static struct cmt_entry *cmt_insert(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa ppa,
				    bool dirty, uint64_t *stime)
{
	struct dftl_cmt *cmt = &conv_ftl->cmt;
	uint32_t tp_idx = lpn / entries_per_tp(conv_ftl);
	struct tp_node *node;
	struct cmt_entry *e;

retry:
	node = node_lookup(cmt, tp_idx);
	/* 기존 노드면 evict 대상에서 보호하기 위해 MRU로 이동 */
	if (node)
		list_move(&node->lru, &cmt->node_lru);

	/* entry 풀이 가득 차면 LRU tail TPnode에서 entry 하나를 evict.
	 * node_size <= entry_size 이므로 entry 검사가 node 풀 검사도 포괄.
	 * 노드가 1개뿐인 degenerate 케이스에선 우리 노드가 evict될 수 있으므로 retry. */
	if (cmt->entry_size >= cmt->entry_capacity) {
		cmt_evict_entry(conv_ftl, stime);
		goto retry;
	}

	if (!node) {
		node = list_first_entry(&cmt->node_free_list, struct tp_node, lru);
		list_del(&node->lru);
		node->tp_idx = tp_idx;
		node->entry_count = 0;
		node->dirty_count = 0;
		INIT_LIST_HEAD(&node->entries);
		hash_add(cmt->node_ht, &node->hnode, tp_idx);
		list_add(&node->lru, &cmt->node_lru);
		cmt->node_size++;
	}

	e = list_first_entry(&cmt->entry_free_list, struct cmt_entry, sibling);
	list_del(&e->sibling);
	e->lpn = lpn;
	e->ppa = ppa;
	e->dirty = dirty;
	e->parent = node;
	hash_add(cmt->entry_ht, &e->hnode, lpn);
	list_add(&e->sibling, &node->entries);
	node->entry_count++;
	if (dirty)
		node->dirty_count++;
	cmt->entry_size++;
	return e;
}

static struct ppa tp_load(struct conv_ftl *conv_ftl, uint64_t lpn, uint64_t *stime)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t ents_per_tp = entries_per_tp(conv_ftl);
	uint32_t tp_idx = lpn / ents_per_tp;
	uint32_t entry_idx = lpn % ents_per_tp;
	struct ppa tp_ppa = conv_ftl->gtd[tp_idx];
	struct ppa *tp_buf;
	struct nand_cmd rd;
	uint64_t nsecs_completed, nsecs_latest = *stime;

	atomic64_inc(&tp_loads);
	/* 이 translation page가 아직 NAND에 없으면 UNMAPPED 반환 */
	if (!mapped_ppa(&tp_ppa)) {
		struct ppa r;
		r.ppa = UNMAPPED_PPA;
		return r;
	}

	/* NAND read 지연 시뮬레이션 */
	rd = (struct nand_cmd){
		.type = USER_IO,
		.cmd = NAND_READ,
		.stime = nsecs_latest,
		.xfer_size = spp->pgsz,
		.interleave_pci_dma = false,
		.ppa = &tp_ppa,
	};
	nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &rd);
	nsecs_latest = max(nsecs_completed, nsecs_latest);
	*stime = nsecs_latest;

	/* 전용 버퍼에서 tp_idx 슬롯의 해당 entry 읽기 */
	tp_buf = (struct ppa *)(conv_ftl->tp_map + (size_t)tp_idx * spp->pgsz);
	return tp_buf[entry_idx];
}

/*
 * Try to answer a mapping lookup from the learned-index model instead of
 * reading the translation page. The prediction is verified against the
 * in-memory reverse map (no NAND access), so a wrong model never corrupts data.
 * Returns true and sets *out on a verified hit.
 */
static bool model_predict(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *out)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t tp_idx = lpn / entries_per_tp(conv_ftl);
	struct lr_node *node = &conv_ftl->lr_nodes[tp_idx];
	uint64_t pred_lpn, pred, pred_pgidx;
	struct ppa ppa;
	int j, seg = -1;

	atomic64_inc(&model_uses);

	if (lpn < node->start_lpn)
		return false;
	pred_lpn = lpn - node->start_lpn;

	for (j = 0; j < LR_MAX_INTERVALS; j++) {
		if (pred_lpn <= node->brks[j].key) {
			seg = j;
			break;
		}
	}
	if (seg < 0)
		return false;

	pred = lr_predict(pred_lpn, node->brks[seg].w_fp, node->brks[seg].b_fp);
	pred_pgidx = node->start_ppa + pred;
	if (pred_pgidx >= spp->tt_pgs)
		return false;

	ppa = pgidx2ppa(conv_ftl, pred_pgidx);
	if (get_rmap_ent(conv_ftl, &ppa) == lpn) {
		*out = ppa;
		atomic64_inc(&model_hits);
		return true;
	}
	return false;
}

static struct ppa dftl_get_ppa(struct conv_ftl *conv_ftl, uint64_t lpn, uint64_t *stime)
{
	struct cmt_entry *e = cmt_lookup(&conv_ftl->cmt, lpn);
	uint32_t ents = entries_per_tp(conv_ftl);
	struct ppa *tp_buf =
		(struct ppa *)(conv_ftl->tp_map + (size_t)(lpn / ents) * conv_ftl->ssd->sp.pgsz);
	struct ppa ppa;

	if (e) {
		/* ppa 정답은 항상 tp_map에서 읽는다(write-through라 늘 최신).
		 * CMT hit이라고 e->ppa를 쓰지 않는다 — hit은 단지 "캐시에 있으니
		 * translation page를 NAND에서 안 읽어도 된다(read 지연 면제)"는 의미일 뿐. */
		return tp_buf[lpn % ents];
	}

	/* CMT miss: try the learned-index model before paying a translation read.
	 * Gated by LEARNED_INDEX_ENABLE so a perf run can A/B learnedFTL vs plain DFTL. */
#if LEARNED_INDEX_ENABLE
	{
		uint32_t tp_idx = lpn / ents;
		struct ppa pred;

		if (conv_ftl->bitmaps[lpn] && conv_ftl->lr_nodes[tp_idx].u &&
		    model_predict(conv_ftl, lpn, &pred))
			return pred;
	}
#endif

	/* prediction unavailable/failed: read translation page and cache it */
	ppa = tp_load(conv_ftl, lpn, stime);
	cmt_insert(conv_ftl, lpn, ppa, false, stime);

	/* cmt_insert() may evict a dirty entry and flush its TPnode; that writeback
	 * (tp_flush_idx)
	 * consumes a write credit which can trigger a foreground GC *re-entrantly*.
	 * Such a GC can relocate THIS lpn and erase the page tp_load() just returned,
	 * leaving `ppa` stale (now PG_FREE). tp_map is write-through and GC keeps it
	 * current, so re-read it for the answer — otherwise conv_write would call
	 * mark_page_invalid() on a freed page and trip its PG_VALID assert. */
	return tp_buf[lpn % ents];
}

/*
 * LearnedFTL Sequential Initialization (HPCA'24, §IV), steps ②③④.
 * Install the accumulated contiguous host-write run as a y=x model for its
 * translation page, but only if it covers more LPNs than the existing model
 * (keep-longer, step ④ — applied uniformly to SI and GC training). The run's
 * LPNs and pgidxs are contiguous (allocation-order assignment), so the mapping
 * is exactly pgidx = start_ppa + (lpn - start_lpn): a single slope-1 segment.
 */
static void si_flush_run(struct conv_ftl *conv_ftl)
{
	uint32_t ents = entries_per_tp(conv_ftl);
	uint64_t s = conv_ftl->si_run_start_lpn;
	uint64_t p = conv_ftl->si_run_start_pgidx;
	uint32_t L = conv_ftl->si_run_len;
	struct lr_node *node;
	uint64_t i;
	int j;

	if (L == 0)
		return;
	conv_ftl->si_run_len = 0;

	node = &conv_ftl->lr_nodes[s / ents];
	if (L <= node->cover_len) /* step ③④: existing model is at least as long */
		return;

	/* step ②: one y=x segment (w=1, b=0), normalized to (s -> p) */
	node->start_lpn = s;
	node->start_ppa = p;
	node->brks[0].w_fp = 1LL << LR_FP_SHIFT;
	node->brks[0].b_fp = 0;
	node->brks[0].key = L - 1;
	node->brks[0].valid_cnt = L;
	for (j = 1; j < LR_MAX_INTERVALS; j++) {
		node->brks[j].w_fp = 0;
		node->brks[j].b_fp = 0;
		node->brks[j].key = L - 1; /* no coverage past the run */
		node->brks[j].valid_cnt = 0;
	}
	node->cover_len = L;
	node->u = 1;
	for (i = s; i < s + L; i++)
		conv_ftl->bitmaps[i] = 1;
	atomic64_inc(&si_installs);
}

/*
 * SI step ① + run accumulation, called once per host-written LPN. A host write
 * invalidates any prior exact-prediction for this LPN (clear its bitmap bit, and
 * keep cover_len = live bit count so keep-longer stays honest). If this write
 * continues the current run (contiguous LPN and pgidx, same translation page) the
 * run grows; otherwise the run is flushed and a new one starts here.
 */
static void si_update(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *ppa)
{
	uint32_t ents = entries_per_tp(conv_ftl);
	uint64_t pgidx = ppa2pgidx(conv_ftl, ppa);

	if (conv_ftl->bitmaps[lpn]) {
		conv_ftl->bitmaps[lpn] = 0;
		if (conv_ftl->lr_nodes[lpn / ents].cover_len)
			conv_ftl->lr_nodes[lpn / ents].cover_len--;
	}

	if (conv_ftl->si_run_len && lpn == conv_ftl->si_run_start_lpn + conv_ftl->si_run_len &&
	    pgidx == conv_ftl->si_run_start_pgidx + conv_ftl->si_run_len &&
	    lpn / ents == conv_ftl->si_run_start_lpn / ents) {
		conv_ftl->si_run_len++;
		return;
	}

	si_flush_run(conv_ftl);
	conv_ftl->si_run_start_lpn = lpn;
	conv_ftl->si_run_start_pgidx = pgidx;
	conv_ftl->si_run_len = 1;
}

static void dftl_set_ppa(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa ppa, uint64_t *stime)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t ents = entries_per_tp(conv_ftl);
	struct ppa *tp_buf = (struct ppa *)(conv_ftl->tp_map + (size_t)(lpn / ents) * spp->pgsz);
	struct cmt_entry *e;

	/* SI step ① + contiguous-run accumulation for sequential initialization */
	si_update(conv_ftl, lpn, &ppa);

	/* write-through: tp_map이 항상 최신인 매핑 정답지가 된다.
	 * 이후 CMT dirty/writeback은 정합성이 아니라 순수 NAND write latency 모델용. */
	tp_buf[lpn % ents] = ppa;

	e = cmt_lookup(&conv_ftl->cmt, lpn);
	if (e) {
		e->ppa = ppa;
		if (!e->dirty) {
			e->dirty = true;
			e->parent->dirty_count++;
		}
		return;
	}

	/* CMT miss: TP를 읽어올 필요 없이 바로 dirty로 삽입 */
	cmt_insert(conv_ftl, lpn, ppa, true, stime);
}

/*
 * GC 재배치 전용 매핑 갱신. tp_map(정답)을 write-through로 갱신하기만 한다.
 * cmt_insert/evict를 절대 호출하지 않아 GC 도중의 writeback cascade(=free line
 * 고갈의 원인)를 차단한다. 재배치된 매핑의 NAND 영속화는 do_gc 종료 시
 * lr_train_collected의 tp_idx별 batched flush가 담당한다.
 */
static void dftl_set_ppa_gc(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t ents = entries_per_tp(conv_ftl);
	struct ppa *tp_buf = (struct ppa *)(conv_ftl->tp_map + (size_t)(lpn / ents) * spp->pgsz);

	/* 매핑 정답은 tp_map 한 곳뿐이라 write-through만 하면 끝.
	 * read는 CMT hit이어도 tp_map에서 읽으므로 GC가 캐시를 맞춰줄 필요가 없다. */
	tp_buf[lpn % ents] = ppa;

	/* relocated page moved: drop any stale exact-prediction (GC analog of SI step ①).
	 * lr_train_collected re-sets bits for samples it can still predict after the move;
	 * dropping cover_len here lets that retrain win keep-longer once the run has moved. */
	if (conv_ftl->bitmaps[lpn]) {
		conv_ftl->bitmaps[lpn] = 0;
		if (conv_ftl->lr_nodes[lpn / ents].cover_len)
			conv_ftl->lr_nodes[lpn / ents].cover_len--;
	}
}

static void init_rmap(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->rmap = vmalloc(sizeof(uint64_t) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		conv_ftl->rmap[i] = INVALID_LPN;
	}
}

static void remove_rmap(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->rmap);
}

static void conv_init_ftl(struct conv_ftl *conv_ftl, struct convparams *cpp, struct ssd *ssd)
{
	/*copy convparams*/
	conv_ftl->cp = *cpp;

	conv_ftl->ssd = ssd;

	/* initialize maptbl */
	//init_maptbl(conv_ftl); // mapping table
	init_dftl(conv_ftl);

	/* initialize rmap */
	init_rmap(conv_ftl); // reverse mapping table (?)

	/* initialize all the lines */
	init_lines(conv_ftl);

	/* initialize write pointers LAZILY: record each frontier's owning group but grab no
	 * line yet (curline == NULL); ensure_write_pointer() grabs one on first write. Under
	 * group==line geometry num_groups == num_lines, so eager init would pin every line. */
	{
		uint32_t g;

		for (g = 0; g < conv_ftl->num_groups; g++)
			conv_ftl->group_wp[g] =
				(struct write_pointer){ .curline = NULL, .group = (int)g };
	}
	conv_ftl->gc_wp = (struct write_pointer){ .curline = NULL, .group = GROUP_NONE };
	conv_ftl->trans_wp = (struct write_pointer){ .curline = NULL, .group = GROUP_NONE };

	init_write_flow_control(conv_ftl);

	NVMEV_INFO("Init FTL instance with %d channels (%ld pages)\n", conv_ftl->ssd->sp.nchs,
		   conv_ftl->ssd->sp.tt_pgs);

	return;
}

static void conv_remove_ftl(struct conv_ftl *conv_ftl)
{
	dftl_group_stats_check(conv_ftl); /* before rmap/lines teardown */
	remove_lines(conv_ftl);
	remove_rmap(conv_ftl);
	//remove_maptbl(conv_ftl);
	remove_dftl(conv_ftl);
}

static void conv_init_params(struct convparams *cpp)
{
	cpp->op_area_pcent = OP_AREA_PERCENT;
	/* host write 1 + GC 한 패스 최악 소비(재배치 vpc + TP batched flush K ≈ 2라인)를
	 * 흡수하기 위한 예약. D-FTL은 GC가 TP write를 추가로 발생시키므로 conv 원본의
	 * 2줄로는 부족해 free line이 고갈된다(NULL deref). */
	cpp->gc_thres_lines = 4;
	cpp->gc_thres_lines_high = 4;
	cpp->enable_gc_delay = 1;
	cpp->pba_pcent = (int)((1 + cpp->op_area_pcent) * 100);
}

void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size, void *mapped_addr,
			 uint32_t cpu_nr_dispatcher)
{
	struct ssdparams spp;
	struct convparams cpp;
	struct conv_ftl *conv_ftls;
	struct ssd *ssd;
	uint32_t i;
	const uint32_t nr_parts = SSD_PARTITIONS;

	ssd_init_params(&spp, size, nr_parts);
	conv_init_params(&cpp);

	conv_ftls = kmalloc(sizeof(struct conv_ftl) * nr_parts, GFP_KERNEL);

	for (i = 0; i < nr_parts; i++) {
		ssd = kmalloc(sizeof(struct ssd), GFP_KERNEL);
		ssd_init(ssd, &spp, cpu_nr_dispatcher);
		conv_init_ftl(&conv_ftls[i], &cpp, ssd);
	}

	/* PCIe, Write buffer are shared by all instances*/
	for (i = 1; i < nr_parts; i++) {
		kfree(conv_ftls[i].ssd->pcie->perf_model);
		kfree(conv_ftls[i].ssd->pcie);
		kfree(conv_ftls[i].ssd->write_buffer);

		conv_ftls[i].ssd->pcie = conv_ftls[0].ssd->pcie;
		conv_ftls[i].ssd->write_buffer = conv_ftls[0].ssd->write_buffer;
	}

	ns->id = id;
	ns->csi = NVME_CSI_NVM;
	ns->nr_parts = nr_parts;
	ns->ftls = (void *)conv_ftls;
	ns->size = (uint64_t)((size * 100) / cpp.pba_pcent);
	ns->mapped = mapped_addr;
	/*register io command handler*/
	ns->proc_io_cmd = conv_proc_nvme_io_cmd;

	NVMEV_INFO("FTL physical space: %lld, logical space: %lld (physical/logical * 100 = %d)\n",
		   size, ns->size, cpp.pba_pcent);

	return;
}

void conv_remove_namespace(struct nvmev_ns *ns)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	const uint32_t nr_parts = SSD_PARTITIONS;
	uint32_t i;

	/* PCIe, Write buffer are shared by all instances*/
	for (i = 1; i < nr_parts; i++) {
		/*
		 * These were freed from conv_init_namespace() already.
		 * Mark these NULL so that ssd_remove() skips it.
		 */
		conv_ftls[i].ssd->pcie = NULL;
		conv_ftls[i].ssd->write_buffer = NULL;
	}

	for (i = 0; i < nr_parts; i++) {
		conv_remove_ftl(&conv_ftls[i]);
		ssd_remove(conv_ftls[i].ssd);
		kfree(conv_ftls[i].ssd);
	}

	kfree(conv_ftls);
	ns->ftls = NULL;
}

static inline bool valid_ppa(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int ch = ppa->g.ch;
	int lun = ppa->g.lun;
	int pl = ppa->g.pl;
	int blk = ppa->g.blk;
	int pg = ppa->g.pg;
	//int sec = ppa->g.sec;

	if (ch < 0 || ch >= spp->nchs)
		return false;
	if (lun < 0 || lun >= spp->luns_per_ch)
		return false;
	if (pl < 0 || pl >= spp->pls_per_lun)
		return false;
	if (blk < 0 || blk >= spp->blks_per_pl)
		return false;
	if (pg < 0 || pg >= spp->pgs_per_blk)
		return false;

	return true;
}

static inline bool valid_lpn(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return (lpn < conv_ftl->ssd->sp.tt_pgs);
}

static inline bool mapped_ppa(struct ppa *ppa)
{
	return !(ppa->ppa == UNMAPPED_PPA);
}

static inline struct line *get_line(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	return &(conv_ftl->lm.lines[ppa->g.blk]);
}

/* update SSD status about one page from PG_VALID -> PG_VALID */
static void mark_page_invalid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	bool was_full_line = false;
	struct line *line;

	/* update corresponding page status */
	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_VALID);
	pg->status = PG_INVALID;

	/* update corresponding block status */
	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->ipc >= 0 && blk->ipc < spp->pgs_per_blk);
	blk->ipc++;
	NVMEV_ASSERT(blk->vpc > 0 && blk->vpc <= spp->pgs_per_blk);
	blk->vpc--;

	/* update corresponding line status */
	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->ipc >= 0 && line->ipc < spp->pgs_per_line);
	if (line->vpc == spp->pgs_per_line) {
		NVMEV_ASSERT(line->ipc == 0);
		was_full_line = true;
	}
	line->ipc++;
	NVMEV_ASSERT(line->vpc > 0 && line->vpc <= spp->pgs_per_line);
	/* Adjust the position of the victime line in the pq under over-writes */
	if (line->pos) {
		/* Note that line->vpc will be updated by this call */
		pqueue_change_priority(lm->victim_line_pq, line->vpc - 1, line);
	} else {
		line->vpc--;
	}

	if (was_full_line) {
		/* move line: "full" -> "victim" */
		list_del_init(&line->entry);
		lm->full_line_cnt--;
		pqueue_insert(lm->victim_line_pq, line);
		lm->victim_line_cnt++;
	}

	/* group stats: DATA page leaving valid (rmap still holds lpn here; the
	 * caller clears it afterwards). valid_pages is keyed by the lpn's group;
	 * invalid_pages by the line's owning group (== sum of owned-line ipc), so a
	 * group's reclaimable garbage is well-defined for victim selection. */
	{
		uint64_t lpn = get_rmap_ent(conv_ftl, ppa);

		if (lpn < TRANS_LPN_BASE) {
			conv_ftl->groups[group_of(conv_ftl, lpn)].valid_pages--;
			if (line->group != GROUP_NONE)
				conv_ftl->groups[line->group].invalid_pages++;
		}
	}
}

static void mark_page_valid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	struct line *line;

	/* update page status */
	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_FREE);
	pg->status = PG_VALID;

	/* update corresponding block status */
	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->vpc >= 0 && blk->vpc < spp->pgs_per_blk);
	blk->vpc++;

	/* update corresponding line status */
	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->vpc >= 0 && line->vpc < spp->pgs_per_line);
	line->vpc++;

	/* group stats: DATA page becoming valid (callers set rmap before this) */
	{
		uint64_t lpn = get_rmap_ent(conv_ftl, ppa);

		if (lpn < TRANS_LPN_BASE)
			conv_ftl->groups[group_of(conv_ftl, lpn)].valid_pages++;
	}
}

static void mark_block_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = get_blk(conv_ftl->ssd, ppa);
	struct nand_page *pg = NULL;
	int i;

	for (i = 0; i < spp->pgs_per_blk; i++) {
		struct ppa page_ppa = *ppa;

		page_ppa.g.pg = i;
		pg = &blk->pg[i];
		NVMEV_ASSERT(pg->nsecs == spp->secs_per_pg);

		/* group stats: a live DATA page in this line is being freed. Its
		 * relocated copy already did valid_pages++ in gc_write_page, so this
		 * decrement balances the move; invalid pages keep no rmap link. */
		if (pg->status == PG_VALID) {
			uint64_t lpn = get_rmap_ent(conv_ftl, &page_ppa);

			if (lpn < TRANS_LPN_BASE)
				conv_ftl->groups[group_of(conv_ftl, lpn)].valid_pages--;
		}

		/* reset page status */
		pg->status = PG_FREE;

		/* Erasing a block also clears its OOB, so drop the stale reverse-map
		 * entry for every freed page. Otherwise rmap[pgidx] keeps pointing at
		 * the lpn that used to live here, and the learned index's model_predict()
		 * — which verifies a prediction solely via get_rmap_ent(ppa)==lpn — can
		 * accept a freed page and hand it back to conv_write, tripping
		 * NVMEV_ASSERT(pg->status == PG_VALID) in mark_page_invalid(). */
		set_rmap_ent(conv_ftl, INVALID_LPN, &page_ppa);
	}

	/* reset block status */
	NVMEV_ASSERT(blk->npgs == spp->pgs_per_blk);
	blk->ipc = 0;
	blk->vpc = 0;
	blk->erase_cnt++;
}

static void gc_read_page(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	/* advance conv_ftl status, we don't care about how long it takes */
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz,
			.interleave_pci_dma = false,
			.ppa = ppa,
		};
		ssd_advance_nand(conv_ftl->ssd, &gcr);
	}
}

/* move valid page data (already in DRAM) from victim line to a new page */
static uint64_t gc_write_page(struct conv_ftl *conv_ftl, struct ppa *old_ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct ppa new_ppa;
	uint64_t lpn = get_rmap_ent(conv_ftl, old_ppa);
	/* translation page는 trans frontier로, 데이터는 GC frontier로 재배치 */
	uint32_t io_type = (lpn >= TRANS_LPN_BASE) ? TRANS_IO : GC_IO;

	//NVMEV_ASSERT(valid_lpn(conv_ftl, lpn));
	new_ppa = get_new_page(conv_ftl, __get_wp(conv_ftl, io_type));
	/* update maptbl */
	//set_maptbl_ent(conv_ftl, lpn, &new_ppa);
	/* update rmap */
	set_rmap_ent(conv_ftl, lpn, &new_ppa);

	mark_page_valid(conv_ftl, &new_ppa);

	/* need to advance the write pointer here */
	advance_write_pointer(conv_ftl, __get_wp(conv_ftl, io_type));

	if (lpn >= TRANS_LPN_BASE) {
		/* TP 내용물은 tp_idx 슬롯에 고정이라 복사 불필요, GTD만 갱신 */
		uint32_t tp_idx = (uint32_t)(lpn - TRANS_LPN_BASE);
		conv_ftl->gtd[tp_idx] = new_ppa;
	} else {
		NVMEV_ASSERT(valid_lpn(conv_ftl, lpn));
		/* GC 전용 경로: CMT insert/evict 없이 tp_map만 갱신 (cascade 차단).
		 * 재배치 매핑은 tp_map(write-through)에 반영되고 모델 재학습으로 복원,
		 * NAND 영속화는 trans frontier의 writeback이 담당. */
		dftl_set_ppa_gc(conv_ftl, lpn, new_ppa);

		/* collect (lpn, new pgidx) for learned-index training at GC end */
		if (conv_ftl->gc_train_cnt < spp->pgs_per_line) {
			int c = conv_ftl->gc_train_cnt++;
			conv_ftl->gc_train_lpns[c] = lpn;
			conv_ftl->gc_train_pgidxs[c] = ppa2pgidx(conv_ftl, &new_ppa);
		}
	}

	if (cpp->enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = GC_IO,
			.cmd = NAND_NOP,
			.stime = 0,
			.interleave_pci_dma = false,
			.ppa = &new_ppa,
		};
		if (last_pg_in_wordline(conv_ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}

		ssd_advance_nand(conv_ftl->ssd, &gcw);
	}

	/* advance per-ch gc_endtime as well */
#if 0
	new_ch = get_ch(conv_ftl, &new_ppa);
	new_ch->gc_endtime = new_ch->next_ch_avail_time;

	new_lun = get_lun(conv_ftl, &new_ppa);
	new_lun->gc_endtime = new_lun->next_lun_avail_time;
#endif

	return 0;
}

static struct line *select_victim_line(struct conv_ftl *conv_ftl, bool force)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *victim_line = NULL;

	victim_line = pqueue_peek(lm->victim_line_pq);
	if (!victim_line) {
		return NULL;
	}

	if (!force && (victim_line->vpc > (spp->pgs_per_line / 8))) {
		return NULL;
	}

	pqueue_pop(lm->victim_line_pq);
	victim_line->pos = 0;
	lm->victim_line_cnt--;

	/* victim_line is a danggling node now */
	return victim_line;
}

/* Pick the data group with the most reclaimable invalid pages. Only sealed
 * victim lines (pos != 0) are reclaimable here; open frontier lines can have
 * invalid pages in groups[].invalid_pages, but do_gc cannot select them yet. */
static int pick_most_invalid_group(struct conv_ftl *conv_ftl)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	uint32_t g;
	int best = GROUP_NONE;
	uint32_t best_ipc = 0;

	for (g = 0; g < conv_ftl->num_groups; g++) {
		uint32_t i;
		uint32_t reclaimable_ipc = 0;

		for (i = 0; i < lm->tt_lines; i++) {
			struct line *ln = &lm->lines[i];

			if (ln->pos == 0 || ln->group != (int)g)
				continue;
			reclaimable_ipc += (uint32_t)ln->ipc;
		}

		if (reclaimable_ipc > best_ipc) {
			best_ipc = reclaimable_ipc;
			best = (int)g;
		}
	}
	return best;
}

/* Select the lowest-vpc victim line owned by `group` and detach it from the
 * shared victim pqueue. A victim line is one currently in the pq (pos != 0).
 * Returns NULL if `group` has no (eligible) victim. */
static struct line *select_victim_line_grp(struct conv_ftl *conv_ftl, int group)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *best = NULL;
	uint32_t i;

	for (i = 0; i < lm->tt_lines; i++) {
		struct line *ln = &lm->lines[i];

		if (ln->pos == 0 || ln->group != group) /* not a victim of this group */
			continue;
		if (!best || ln->vpc < best->vpc)
			best = ln;
	}

	/* whole-group GC: return this group's lowest-vpc victim, or NULL when the group has
	 * none left in the victim pq. No vpc cap — do_gc drains the entire target group in
	 * one sweep so its survivors relocate as a single LPN-sorted contiguous run that the
	 * learned models fit; do_gc tops up with global victims for free-list safety. */
	if (!best)
		return NULL;

	pqueue_remove(lm->victim_line_pq, best);
	best->pos = 0;
	lm->victim_line_cnt--;
	return best;
}

/* here ppa identifies the block we want to clean */
static void clean_one_block(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_page *pg_iter = NULL;
	int cnt = 0;
	int pg;

	for (pg = 0; pg < spp->pgs_per_blk; pg++) {
		ppa->g.pg = pg;
		pg_iter = get_pg(conv_ftl->ssd, ppa);
		/* there shouldn't be any free page in victim blocks */
		NVMEV_ASSERT(pg_iter->status != PG_FREE);
		if (pg_iter->status == PG_VALID) {
			gc_read_page(conv_ftl, ppa);
			/* delay the maptbl update until "write" happens */
			gc_write_page(conv_ftl, ppa);
			cnt++;
		}
	}

	NVMEV_ASSERT(get_blk(conv_ftl->ssd, ppa)->vpc == cnt);
}

/*
 * GC relocation helpers — LPN-sorted segment cleaning.
 *
 * The learned index fits lpn -> vPPN, and our pgidx is allocation-ordered, so a run of
 * consecutive LPNs relocated back-to-back lands on consecutive vPPNs (a fittable straight
 * line). To produce those runs, do_gc gathers the victim's data-page LPNs
 * (gc_collect_flashpg), sorts them, and only then relocates via gc_relocate_data — instead
 * of relocating in physical scan order, which scrambled lpn->vPPN and left the model almost
 * no exact matches. (gc_write_page above is the old scan-order relocator, now unused.)
 */

/* Allocate a fresh page on the given frontier for `lpn`, mark it valid, advance the write
 * pointer, and model the NAND program latency. Returns the new ppa. */
static struct ppa gc_alloc_and_write(struct conv_ftl *conv_ftl, uint64_t lpn,
				     struct write_pointer *wp)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct ppa new_ppa;

	/* gc_flag=false: this IS the GC relocation path — must not re-enter do_gc */
	ensure_write_pointer(conv_ftl, wp, false);
	new_ppa = get_new_page(conv_ftl, wp);
	set_rmap_ent(conv_ftl, lpn, &new_ppa);
	mark_page_valid(conv_ftl, &new_ppa);
	advance_write_pointer(conv_ftl, wp);

	if (cpp->enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = GC_IO,
			.cmd = NAND_NOP,
			.stime = 0,
			.interleave_pci_dma = false,
			.ppa = &new_ppa,
		};
		if (last_pg_in_wordline(conv_ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}
		ssd_advance_nand(conv_ftl->ssd, &gcw);
	}
	return new_ppa;
}

/* Relocate one translation page (lpn encodes its tp_idx) to the TRANS frontier; the page
 * content lives in tp_map[tp_idx], so only the GTD entry moves. */
static void gc_relocate_tp(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	struct ppa new_ppa = gc_alloc_and_write(conv_ftl, lpn, &conv_ftl->trans_wp);

	conv_ftl->gtd[(uint32_t)(lpn - TRANS_LPN_BASE)] = new_ppa;
}

/* Relocate one data page back to ITS GROUP's own write frontier (ld-tpftl model): the
 * compacted data stays group-owned (re-GC-able) and group_wp[G] only ever holds group G's
 * LPNs, so strict 1-line-1-group holds with no separate gc_wp / per-group GC frontier. */
static struct ppa gc_relocate_data(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	struct write_pointer *wp = &conv_ftl->group_wp[group_of(conv_ftl, lpn)];
	struct ppa new_ppa = gc_alloc_and_write(conv_ftl, lpn, wp);

	dftl_set_ppa_gc(conv_ftl, lpn, new_ppa);
	return new_ppa;
}

/* Collect phase for one flash page of the victim: account the batched read, relocate any
 * translation pages immediately, and gather data-page LPNs for later LPN-sorted relocation. */
static void gc_collect_flashpg(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct nand_page *pg_iter = NULL;
	int cnt = 0, i = 0;
	struct ppa ppa_copy = *ppa;

	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		/* there shouldn't be any free page in victim blocks */
		NVMEV_ASSERT(pg_iter->status != PG_FREE);
		if (pg_iter->status == PG_VALID)
			cnt++;

		ppa_copy.g.pg++;
	}

	ppa_copy = *ppa;

	if (cnt <= 0)
		return;

	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false,
			.ppa = &ppa_copy,
		};
		ssd_advance_nand(conv_ftl->ssd, &gcr);
	}

	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);

		if (pg_iter->status == PG_VALID) {
			uint64_t lpn = get_rmap_ent(conv_ftl, &ppa_copy);

			if (lpn >= TRANS_LPN_BASE) {
				/* translation page: relocate immediately */
				gc_relocate_tp(conv_ftl, lpn);
			} else {
				/* data page: gather LPN; relocated in sorted order in do_gc */
				NVMEV_ASSERT(valid_lpn(conv_ftl, lpn));
				NVMEV_ASSERT(conv_ftl->gc_train_cnt < conv_ftl->gc_train_cap);
				conv_ftl->gc_train_lpns[conv_ftl->gc_train_cnt++] = lpn;
			}
		}

		ppa_copy.g.pg++;
	}
}

static void mark_line_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *line = get_line(conv_ftl, ppa);
	/* drop this line's invalid pages and owned-line count from its group before reset */
	if (line->group != GROUP_NONE) {
		conv_ftl->groups[line->group].invalid_pages -= line->ipc;
		conv_ftl->groups[line->group].alloc_lines--;
	}
	line->ipc = 0;
	line->vpc = 0;
	line->group = GROUP_NONE; /* released line is no longer owned by any group */
	/* move this line to free line list */
	list_add_tail(&line->entry, &lm->free_line_list);
	lm->free_line_cnt++;
}

/* foreground_gc()
 * -> do_gc()
 *    -> target group 선택
 *    -> victim line 선택
 *    -> valid page scan
 *       -> translation page는 즉시 relocate
 *       -> data page는 LPN만 수집
 *    -> victim block erase
 *    -> victim line free list로 반환
 *    -> data LPN 정렬
 *    -> data page를 LPN 순서로 relocate
 *    -> 새 pgidx 기반 learned index 재학습
 */
static int do_gc(struct conv_ftl *conv_ftl, bool force)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct ppa ppa;
	int flashpg, k;
	int nlines = 0;
	int target;

	conv_ftl->gc_train_cnt = 0;
	conv_ftl->wfc.credits_to_refill = 0;

	/*
	 * Stage A — whole-group sweep (LearnedFTL group GC): target the most-invalid group
	 * and drain ALL of its victim lines in one pass, so its survivors relocate as a
	 * single LPN-sorted contiguous run (Stage B/C) and its models retrain as a unit
	 * (Stage D) — that contiguity is what lifts learned-index coverage. Relocation stays
	 * on the shared GROUP_NONE gc_wp. Re-grouping the relocated data (tagging it with the
	 * swept group so it stays re-GC-able) needs strict 1-line-1-group, which a single
	 * shared frontier can't keep: its curline persists across sweeps, so the next group's
	 * relocation lands in the previous group's tagged line -> GROUP ROUTING VIOLATION.
	 * Proper re-grouping needs per-group gc frontiers (and OP headroom, since slack <
	 * num_groups here). Once the group is drained, top up to at least GC_BATCH_LINES with
	 * global victims (free-list safety). Free-before-rewrite holds: Stage A frees every
	 * victim before Stage C relocates, and relocation (< vpc of the freed lines) consumes
	 * fewer lines than were freed, so a sweep can never exhaust the free list.
	 */
	target = pick_most_invalid_group(conv_ftl);
	/* WA control: only spend a whole-group sweep on a group that has accumulated
	 * enough garbage to be worth compacting; otherwise fall to global lowest-vpc
	 * victims (space-efficient, low write-amplification). */
	if (target != GROUP_NONE &&
	    conv_ftl->groups[target].invalid_pages <
		    (uint32_t)GROUP_GC_MIN_INVALID_LINES * spp->pgs_per_line)
		target = GROUP_NONE;

	for (;;) {
		struct line *victim_line = NULL;

		/* bound a sweep to the gc_train buffer (one group + top-up); any leftover
		 * victim is reclaimed by the next GC pass. Checked before popping so we never
		 * detach a victim we cannot process. */
		if (conv_ftl->gc_train_cnt + spp->pgs_per_line > conv_ftl->gc_train_cap)
			break;

		/* drain the target group's victims first (whole-group contiguity); once it is
		 * empty, top up with global victims only until the pass has freed a batch, so a
		 * near-clean target can't starve the free list (stale curline -> PG_FREE). */
		if (target != GROUP_NONE)
			victim_line = select_victim_line_grp(conv_ftl, target);
		if (!victim_line) {
			if (nlines >= GC_BATCH_LINES)
				break;
			victim_line = select_victim_line(conv_ftl, force);
			NVMEV_DEBUG_VERBOSE("select victim_line(fallback)\n");
		}
		if (!victim_line)
			break;

		ppa.g.blk = victim_line->id;
		conv_ftl->wfc.credits_to_refill += victim_line->ipc;
		atomic64_inc(&gc_runs);
		nlines++;

		for (flashpg = 0; flashpg < spp->flashpgs_per_blk; flashpg++) {
			int ch, lun;

			ppa.g.pg = flashpg * spp->pgs_per_flashpg;
			for (ch = 0; ch < spp->nchs; ch++) {
				for (lun = 0; lun < spp->luns_per_ch; lun++) {
					struct nand_lun *lunp;

					ppa.g.ch = ch;
					ppa.g.lun = lun;
					ppa.g.pl = 0;
					lunp = get_lun(conv_ftl->ssd, &ppa);
					gc_collect_flashpg(conv_ftl, &ppa);

					if (flashpg == (spp->flashpgs_per_blk - 1)) {
						struct convparams *cpp = &conv_ftl->cp;

						mark_block_free(conv_ftl, &ppa);

						if (cpp->enable_gc_delay) {
							struct nand_cmd gce = {
								.type = GC_IO,
								.cmd = NAND_ERASE,
								.stime = 0,
								.interleave_pci_dma = false,
								.ppa = &ppa,
							};
							ssd_advance_nand(conv_ftl->ssd, &gce);
						}

						lunp->gc_endtime = lunp->next_lun_avail_time;
					}
				}
			}
		}

		/* free this victim now so Stage C can relocate into its reclaimed space */
		mark_line_free(conv_ftl, &ppa);
	}

	if (nlines == 0)
		return -1;

	if (target != GROUP_NONE)
		atomic64_inc(&gc_group_targeted);
	else
		atomic64_inc(&gc_fallback);

	/* pace GC: ensure at least a line's worth of write credits before the next
	 * foreground_gc check, so small-ipc victims don't trigger a re-entry storm */
	if (conv_ftl->wfc.credits_to_refill < (int64_t)spp->pgs_per_line)
		conv_ftl->wfc.credits_to_refill = (int64_t)spp->pgs_per_line;

	/*
	 * Stage B — sort the whole batch's data LPNs ascending. Relocating in this order makes
	 * each translation-page group a monotonically increasing vPPN run, which the learned
	 * index fits with far more exact matches. gc_train_pgidxs is (re)written in Stage C, so
	 * its values here are a don't-care that the pair-sort merely carries along.
	 */
	if (conv_ftl->gc_train_cnt > 1)
		lr_sort_pairs(conv_ftl->gc_train_lpns, conv_ftl->gc_train_pgidxs, 0,
			      conv_ftl->gc_train_cnt - 1);

	/* Stage C — relocate data pages in LPN order; record (lpn, new vPPN) for training. */
	for (k = 0; k < conv_ftl->gc_train_cnt; k++) {
		struct ppa new_ppa = gc_relocate_data(conv_ftl, conv_ftl->gc_train_lpns[k]);

		conv_ftl->gc_train_pgidxs[k] = ppa2pgidx(conv_ftl, &new_ppa);
	}

	/* Stage D — train learned-index models on the LPN-sorted (cross-line) runs. */
	lr_train_collected(conv_ftl);

	return 0;
}

static void foreground_gc(struct conv_ftl *conv_ftl)
{
	/* GC trigger #1 only (free flash blocks low): reclaim the most-invalid group (or global
	 * lowest-vpc victims) in do_gc. Trigger #2 (per-group cumulative-block threshold) was
	 * dropped — nvmevirt's bounded channel timing model can't absorb proactive whole-group GC
	 * bursts under WEC=1, and at the near-full utilization where the learned index is measured
	 * free-line pressure already drives GC (and model retraining) plenty. */
	if (should_gc_high(conv_ftl))
		do_gc(conv_ftl, true);
}

static bool is_same_flash_page(struct conv_ftl *conv_ftl, struct ppa ppa1, struct ppa ppa2)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t ppa1_page = ppa1.g.pg / spp->pgs_per_flashpg;
	uint32_t ppa2_page = ppa2.g.pg / spp->pgs_per_flashpg;

	return (ppa1.h.blk_in_ssd == ppa2.h.blk_in_ssd) && (ppa1_page == ppa2_page);
}

static bool conv_read(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];
	/* spp are shared by all instances*/
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn;
	uint64_t nsecs_start = req->nsecs_start;
	uint64_t nsecs_completed, nsecs_latest = nsecs_start;
	uint32_t xfer_size, i;
	uint32_t nr_parts = ns->nr_parts;

	struct ppa prev_ppa;
	struct nand_cmd srd = {
		.type = USER_IO,
		.cmd = NAND_READ,
		.stime = nsecs_start,
		.interleave_pci_dma = true,
	};

	NVMEV_ASSERT(conv_ftls);
	NVMEV_DEBUG_VERBOSE("%s: start_lpn=%lld, len=%lld, end_lpn=%lld", __func__, start_lpn,
			    nr_lba, end_lpn);
	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn passed FTL range (start_lpn=%lld > tt_pgs=%ld)\n", __func__,
			    start_lpn, spp->tt_pgs);
		return false;
	}

	if (LBA_TO_BYTE(nr_lba) <= (KB(4) * nr_parts)) {
		srd.stime += spp->fw_4kb_rd_lat;
	} else {
		srd.stime += spp->fw_rd_lat;
	}

	for (i = 0; (i < nr_parts) && (start_lpn <= end_lpn); i++, start_lpn++) {
		conv_ftl = &conv_ftls[start_lpn % nr_parts];
		xfer_size = 0;
		//prev_ppa = get_maptbl_ent(conv_ftl, start_lpn / nr_parts);
		prev_ppa = dftl_get_ppa(conv_ftl, start_lpn / nr_parts, &srd.stime);

		/* normal IO read path */
		for (lpn = start_lpn; lpn <= end_lpn; lpn += nr_parts) {
			uint64_t local_lpn;
			struct ppa cur_ppa;

			local_lpn = lpn / nr_parts;
			//cur_ppa = get_maptbl_ent(conv_ftl, local_lpn);
			cur_ppa = dftl_get_ppa(conv_ftl, local_lpn, &srd.stime);

			if (!mapped_ppa(&cur_ppa) || !valid_ppa(conv_ftl, &cur_ppa)) {
				NVMEV_DEBUG_VERBOSE("lpn 0x%llx not mapped to valid ppa\n",
						    local_lpn);
				NVMEV_DEBUG_VERBOSE("Invalid ppa,ch:%d,lun:%d,blk:%d,pl:%d,pg:%d\n",
						    cur_ppa.g.ch, cur_ppa.g.lun, cur_ppa.g.blk,
						    cur_ppa.g.pl, cur_ppa.g.pg);
				continue;
			}

			// aggregate read io in same flash page
			if (mapped_ppa(&prev_ppa) &&
			    is_same_flash_page(conv_ftl, cur_ppa, prev_ppa)) {
				xfer_size += spp->pgsz;
				continue;
			}

			if (xfer_size > 0) {
				srd.xfer_size = xfer_size;
				srd.ppa = &prev_ppa;
				nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &srd);
				nsecs_latest = max(nsecs_completed, nsecs_latest);
			}

			xfer_size = spp->pgsz;
			prev_ppa = cur_ppa;
		}

		// issue remaining io
		if (xfer_size > 0) {
			srd.xfer_size = xfer_size;
			srd.ppa = &prev_ppa;
			nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &srd);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
		}
	}

	ret->nsecs_target = nsecs_latest;
	ret->status = NVME_SC_SUCCESS;
	return true;
}

static bool conv_write(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];

	/* wbuf and spp are shared by all instances */
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct buffer *wbuf = conv_ftl->ssd->write_buffer;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;

	uint64_t lpn;
	uint32_t nr_parts = ns->nr_parts;

	uint64_t nsecs_latest;
	uint64_t nsecs_xfer_completed;
	uint32_t allocated_buf_size;

	struct nand_cmd swr = {
		.type = USER_IO,
		.cmd = NAND_WRITE,
		.interleave_pci_dma = false,
		.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg,
	};

	NVMEV_DEBUG_VERBOSE("%s: start_lpn=%lld, len=%lld, end_lpn=%lld", __func__, start_lpn,
			    nr_lba, end_lpn);
	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn passed FTL range (start_lpn=%lld > tt_pgs=%ld)\n", __func__,
			    start_lpn, spp->tt_pgs);
		return false;
	}

	allocated_buf_size = buffer_allocate(wbuf, LBA_TO_BYTE(nr_lba));
	if (allocated_buf_size < LBA_TO_BYTE(nr_lba))
		return false;

	nsecs_latest =
		ssd_advance_write_buffer(conv_ftl->ssd, req->nsecs_start, LBA_TO_BYTE(nr_lba));
	nsecs_xfer_completed = nsecs_latest;

	swr.stime = nsecs_latest;

	for (lpn = start_lpn; lpn <= end_lpn; lpn++) {
		uint64_t local_lpn;
		uint64_t nsecs_completed = 0;
		struct ppa ppa;

		conv_ftl = &conv_ftls[lpn % nr_parts];
		local_lpn = lpn / nr_parts;
		//ppa = get_maptbl_ent(
		//	conv_ftl, local_lpn); // Check whether the given LPN has been written before
		ppa = dftl_get_ppa(conv_ftl, local_lpn, &swr.stime);
		if (mapped_ppa(&ppa)) {
			/* update old page information first */
			mark_page_invalid(conv_ftl, &ppa);
			set_rmap_ent(conv_ftl, INVALID_LPN, &ppa);
			NVMEV_DEBUG("%s: %lld is invalid, ", __func__, ppa2pgidx(conv_ftl, &ppa));
		}

		/* new write — route to this lpn's group frontier (group-owned alloc) */
		{
			struct write_pointer *uwp =
				&conv_ftl->group_wp[group_of(conv_ftl, local_lpn)];

			ensure_write_pointer(conv_ftl, uwp, true);
			ppa = get_new_page(conv_ftl, uwp);
			/* update maptbl */
			//set_maptbl_ent(conv_ftl, local_lpn, &ppa);
			dftl_set_ppa(conv_ftl, local_lpn, ppa, &swr.stime);
			NVMEV_DEBUG("%s: got new ppa %lld, ", __func__, ppa2pgidx(conv_ftl, &ppa));
			/* update rmap */
			set_rmap_ent(conv_ftl, local_lpn, &ppa);

			mark_page_valid(conv_ftl, &ppa);

			/* need to advance the write pointer here */
			advance_write_pointer(conv_ftl, uwp);
		}

		/* Aggregate write io in flash page */
		if (last_pg_in_wordline(conv_ftl, &ppa)) {
			swr.ppa = &ppa;

			nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &swr);
			nsecs_latest = max(nsecs_completed, nsecs_latest);

			schedule_internal_operation(req->sq_id, nsecs_completed, wbuf,
						    spp->pgs_per_oneshotpg * spp->pgsz);
		}

		atomic64_inc(&host_writes);
		consume_write_credit(conv_ftl);
		check_and_refill_write_credit(conv_ftl);
	}

	if ((cmd->rw.control & NVME_RW_FUA) || (spp->write_early_completion == 0)) {
		/* Wait all flash operations */
		ret->nsecs_target = nsecs_latest;
	} else {
		/* Early completion */
		ret->nsecs_target = nsecs_xfer_completed;
	}
	ret->status = NVME_SC_SUCCESS;

	return true;
}

static void conv_flush(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	uint64_t start, latest;
	uint32_t i;
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;

	start = local_clock();
	latest = start;
	for (i = 0; i < ns->nr_parts; i++) {
		latest = max(latest, ssd_next_idle_time(conv_ftls[i].ssd));
	}

	NVMEV_DEBUG_VERBOSE("%s: latency=%llu\n", __func__, latest - start);

	ret->status = NVME_SC_SUCCESS;
	ret->nsecs_target = latest;
	return;
}

bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct nvme_command *cmd = req->cmd;

	NVMEV_ASSERT(ns->csi == NVME_CSI_NVM);

	switch (cmd->common.opcode) {
	case nvme_cmd_write:
		if (!conv_write(ns, req, ret))
			return false;
		break;
	case nvme_cmd_read:
		if (!conv_read(ns, req, ret))
			return false;
		break;
	case nvme_cmd_flush:
		conv_flush(ns, req, ret);
		break;
	default:
		NVMEV_ERROR("%s: command not implemented: %s (0x%x)\n", __func__,
			    nvme_opcode_string(cmd->common.opcode), cmd->common.opcode);
		break;
	}

	return true;
}
