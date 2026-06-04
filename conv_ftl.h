// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include <linux/list.h>
#include <linux/hashtable.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "ssd.h"

#define CMT_CAPACITY 64
#define CMT_ENTRY_HASH_BITS 6
#define CMT_ENTRY_HASH_SIZE (1 << CMT_ENTRY_HASH_BITS)
#define CMT_NODE_HASH_BITS 6
#define CMT_NODE_HASH_SIZE (1 << CMT_NODE_HASH_BITS)
#define TRANS_LPN_BASE (INVALID_LPN - (1ULL << 32))

/* learned index (LearnedFTL, HPCA'24) parameters */
#define LR_MAX_INTERVALS 8	/* piecewise segments per translation page model */
#define LR_TRAIN_THRESHOLD 30	/* min samples in a TP group to train a model */
#define LR_FP_SHIFT 16		/* fixed-point fractional bits (kernel has no FPU) */
#define GC_BATCH_LINES 4	/* victim lines batched per do_gc for segment cleaning */

/* GTD-entry grouping for group-granular GC (LearnedFTL). A group owns
 * TP_PER_GROUP consecutive translation pages; group_of(lpn) =
 * (lpn / entries_per_tp) / TP_PER_GROUP. */
#define TP_PER_GROUP 64
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
	uint32_t write_credits;
	uint32_t credits_to_refill;
};

/* per-group accounting for group-granular GC (LearnedFTL).
 * Phase 1 tracks only valid_pages (live DATA pages whose lpn maps into the
 * group); invalid/used-line accounting arrives with group-owned allocation. */
struct gtd_group {
	uint32_t valid_pages;
};

struct conv_ftl {
	struct ssd *ssd;

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

	/* training sample collection during one line GC (size pgs_per_line each) */
	uint64_t *gc_train_lpns;
	uint64_t *gc_train_pgidxs;
	int gc_train_cnt;

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

#endif
