// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/hashtable.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "ssd.h"

#define CMT_CAPACITY 131072
#define CMT_ENTRY_HASH_BITS 17
#define CMT_ENTRY_HASH_SIZE (1 << CMT_ENTRY_HASH_BITS)
#define CMT_NODE_HASH_BITS 10
#define CMT_NODE_HASH_SIZE (1 << CMT_NODE_HASH_BITS)
#define TRANS_LPN_BASE (INVALID_LPN - (1ULL << 32))

/* learned index (LearnedFTL, HPCA'24) parameters */
/* A/B switch: 1 = learned index serves reads (LearnedFTL); 0 = every CMT miss pays
 * the translation read (plain DFTL). Flip + rebuild to isolate the learned index's
 * IOPS contribution. SI/GC training still run when 0 but are unused (and a pure read
 * phase does no training), so the on/off IOPS delta is exactly the read-path benefit. */
#define LEARNED_INDEX_ENABLE 1
/* CMT fill granularity A/B switch (DFTL CMT 관리 단위):
 * 1 = translation-page granular: CMT miss 시 그 TP 1장의 entry 512개를 unmapped 슬롯까지
 *     통째로 캐시하고 evict도 TP node 단위. 같은 TP 안 인접 lpn 재접근의 NAND translation read를
 *     없앤다(DFTL 정석). 0 = entry granular: miss 시 요청 lpn 1개만 캐시/evict(fine-grained).
 * 두 모드 모두 CMT_CAPACITY entry 슬롯이라는 동일 메모리 예산을 공유하므로 fill 단위만
 * 격리 비교된다. LEARNED_INDEX_ENABLE과 직교 — model_predict가 흡수 못 한 miss에만 작동. */
#define CMT_FILL_TP 1
#define LR_MAX_INTERVALS 8 /* piecewise segments per translation page model */
#define LR_TRAIN_THRESHOLD 30 /* min samples in a TP group to train a model */
#define LR_FP_SHIFT 16 /* fixed-point fractional bits (kernel has no FPU) */
#define GC_BATCH_LINES 4 /* victim lines batched per do_gc for segment cleaning */
/* background GC (전용 kthread, gc_cpus 파라미터): */
#define GC_RESERVE_LINES \
	4 /* user 쓰기는 free line이 이 밑으로 내려가면 stall — GC relocation(Stage C)이
	   * frontier를 새로 열 때 쓸 line을 보장 (GC 트리거 thres 32보다 한참 아래) */
#define GC_NAND_BACKLOG_NS \
	(200ULL * 1000 * 1000) /* GC가 쌓아둘 수 있는 미래 NAND 작업 상한(200ms). do_gc 한
	   * 번이 그룹 전체의 NAND op를 wallclock-now로 일괄 발행하므로, 다음 do_gc 전에
	   * lun avail time이 이 밑으로 빠질 때까지 대기 — chmodel credit window(~2.1s)
	   * overflow("No free entry" flood)를 원천 차단하는 pacing */
/* WA control for whole-group GC: a group earns a one-sweep whole-group compaction
 * (contiguous LPN-sorted relocation) only once it holds at least this many lines'
 * worth of invalid pages; below it, do_gc uses efficient global lowest-vpc victims.
 * Raise to cut write-amplification (fewer, riper sweeps) at some learned-index
 * coverage; lower toward 0 for max coverage (every most-invalid group swept). */
#define GROUP_GC_MIN_INVALID_LINES \
	1 /* whole-group sweep only for groups ripe with >=4 lines of garbage; others fall to the
	   * efficient global lowest-vpc fallback (frees near-empty lines, low valid relocation).
	   * Was set to 0 mid-session (force group_gc>0) but that whole-group-sweeps barely-invalid
	   * groups -> relocates mostly-valid data -> WA balloons (saw 2.90 on bug-rw vs ~1.x). */

/* GTD-entry grouping for group-granular GC (LearnedFTL). A group owns
 * TP_PER_GROUP consecutive translation pages; group_of(lpn) =
 * (lpn / entries_per_tp) / TP_PER_GROUP. */
#define TP_PER_GROUP \
	16 /* group = 8 TP = 16MB = exactly 1 line: strict 1-line-1-group, light per-event GC */
#define GROUP_NONE (-1) /* line/wp not owned by a data group (gc/trans frontier) */

/* one piecewise-linear segment: pgidx ~= (w_fp*x + b_fp) >> LR_FP_SHIFT, x = lpn - start_lpn */
struct lr_breakpoint {
	int64_t w_fp; /* slope, fixed-point */
	int64_t b_fp; /* intercept, fixed-point */
	uint32_t key; /* upper bound (inclusive) of normalized lpn for this segment */
	uint32_t valid_cnt; /* #samples this segment predicts exactly */
};

/* per translation-page linear-regression model */
struct lr_node {
	struct lr_breakpoint brks[LR_MAX_INTERVALS];
	uint64_t start_lpn; /* model is relative to this lpn */
	uint64_t start_ppa; /* ... and this pgidx */
	uint32_t cover_len; /* #LPNs this model currently predicts; keep-longer arbiter (SI step ④) */
	uint8_t u; /* 1 if trained/usable */
};

struct tp_node;

struct cmt_entry {
	uint64_t lpn;
	struct ppa ppa;
	bool dirty;
	struct tp_node *parent;
	struct list_head sibling; /* parent->entries when in use, entry_free_list when free */
	struct hlist_node hnode;
};

struct tp_node {
	uint32_t tp_idx;
	uint32_t entry_count;
	uint32_t dirty_count;
	struct list_head entries;
	struct list_head lru; /* node_lru when in use, node_free_list when free */
	struct hlist_node hnode;
};

struct dftl_cmt {
	struct cmt_entry *entry_pool;
	struct tp_node *node_pool;
	struct hlist_head entry_ht[CMT_ENTRY_HASH_SIZE];
	struct hlist_head node_ht[CMT_NODE_HASH_SIZE];
	struct list_head node_lru;
	struct list_head entry_free_list;
	struct list_head node_free_list;
	uint32_t entry_size;
	uint32_t entry_capacity;
	uint32_t node_size;
	uint32_t node_capacity;
};

struct convparams {
	uint32_t gc_thres_lines;
	uint32_t gc_thres_lines_high;
	bool enable_gc_delay;

	double op_area_pcent;
	int pba_pcent; /* (physical space / logical space) * 100*/
};

struct line {
	int id; /* line id, the same as corresponding block id */
	int ipc; /* invalid page count in this line */
	int vpc; /* valid page count in this line */
	int group; /* owning data group, or GROUP_NONE (gc/trans/free line) */
	struct list_head entry;
	/* position in the priority queue for victim lines */
	size_t pos;
};

/* wp: record next write addr */
struct write_pointer {
	struct line *curline;
	uint32_t ch;
	uint32_t lun;
	uint32_t pg;
	uint32_t blk;
	uint32_t pl;
	int group; /* data group this frontier writes for, or GROUP_NONE */
};

struct line_mgmt {
	struct line *lines;

	/* free line list, we only need to maintain a list of blk numbers */
	struct list_head free_line_list;
	pqueue_t *victim_line_pq;
	struct list_head full_line_list;

	uint32_t tt_lines;
	uint32_t free_line_cnt;
	uint32_t victim_line_cnt;
	uint32_t full_line_cnt;
};

struct write_flow_control {
	int64_t write_credits;
	int64_t credits_to_refill;
};

/* per-group accounting for group-granular GC (LearnedFTL).
 * Phase 1 tracks only valid_pages (live DATA pages whose lpn maps into the
 * group); invalid/used-line accounting arrives with group-owned allocation. */
struct gtd_group {
	uint32_t valid_pages; /* live DATA pages whose lpn maps into this group */
	uint32_t invalid_pages; /* invalid DATA pages in this group's lines (== sum of owned-line ipc) */
	uint32_t alloc_lines; /* lines this group currently owns (LearnedFTL cumulative_allocated_blocks) */
};

struct conv_ftl {
	struct ssd *ssd;

	/* serializes FTL metadata (maptbl/tp_map/CMT/rmap/lines/wp/page state) and
	 * this partition's NAND-timing advance between the dispatcher and the
	 * background GC thread. Dispatcher holds it across each per-partition
	 * section of conv_read/conv_write; nested paths (foreground GC, CMT
	 * eviction, TP flush) run under it and must NOT re-acquire. */
	spinlock_t ftl_lock;
	uint32_t part_id; /* partition index (diag logs) */
	/* user 쓰기가 free-line reserve에서 stall 중 (ensure_write_pointer).
	 * GC 스레드는 이게 켜진 파티션의 NAND backlog pacing을 무시(urgent bypass)
	 * — pacing이 starving 파티션을 계속 건너뛰면 stall이 영원히 안 풀린다. */
	bool user_stalled;

	struct convparams cp;
	//struct ppa *maptbl; /* page level mapping table */

	struct ppa *gtd;
	struct dftl_cmt cmt;
	uint32_t num_tp;
	void *tp_map; /* translation page content store, indexed by tp_idx */

	uint32_t num_groups; /* DIV_ROUND_UP(num_tp, TP_PER_GROUP) */
	struct gtd_group *groups; /* size num_groups, indexed by group_of(lpn) */

	struct lr_node *lr_nodes; /* learned-index models, indexed by tp_idx (size num_tp) */
	uint8_t *bitmaps; /* per-lpn: 1 if model predicts it exactly (size tt_pgs) */

	/* sequential-initialization (LearnedFTL): the current contiguous host-write run,
	 * accumulated across writes and flushed as a y=x model when it breaks */
	uint64_t si_run_start_lpn;
	uint64_t si_run_start_pgidx;
	uint32_t si_run_len;

	/* training sample collection during one line GC (size pgs_per_line each) */
	uint64_t *gc_train_lpns;
	uint64_t *gc_train_pgidxs;
	int gc_train_cnt;
	uint32_t gc_train_cap; /* capacity of gc_train_* — one whole group's pages + top-up */

	uint64_t *rmap; /* reverse mapptbl, assume it's stored in OOB */
	struct write_pointer *group_wp; /* per-group user write frontiers, size num_groups */
	struct write_pointer gc_wp;
	struct write_pointer trans_wp; /* translation page writeback frontier */
	struct line_mgmt lm;
	struct write_flow_control wfc;
};

void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size, void *mapped_addr,
			 uint32_t cpu_nr_dispatcher);

void conv_remove_namespace(struct nvmev_ns *ns);

bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret);

/* background GC kthread entry; data = struct nvmev_gc_worker * */
int nvmev_gc_fn(void *data);

/* Read-path hit counters, surfaced/reset via /proc/nvmev/learnedFTL */
void conv_cmt_stat_read(uint64_t *access, uint64_t *hit, uint64_t *miss);
void conv_cmt_stat_reset(void);
/* LR model counters, surfaced alongside CMT stats via /proc/nvmev/learnedFTL */
void conv_lr_stat_read(uint64_t *trains, uint64_t *bits_set, uint64_t *uses,
		       uint64_t *hits, uint64_t *si);

#endif
