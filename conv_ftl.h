// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEVIRT_CONV_FTL_H
#define _NVMEVIRT_CONV_FTL_H

#include <linux/types.h>
#include <linux/list.h>
#include "pqueue/pqueue.h"
#include "ssd_config.h"
#include "ssd.h"

#define CMT_CAPACITY 256  /* number of TP pages to cache (256 × 512 entries = 128K LPN coverage) */
#define TRANS_LPN_BASE (INVALID_LPN - (1ULL << 32))

struct dftl_cmt {
	uint32_t size;     /* number of TP pages currently cached */
	uint32_t capacity; /* max TP pages in cache */
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

struct conv_ftl {
	struct ssd *ssd;

	struct convparams cp;
	//struct ppa *maptbl; /* page level mapping table */

	struct ppa *gtd;
	struct ppa *tp_data;            /* TP page content: [tp_idx * entries_per_tp + entry_idx] */
	bool *tp_dirty;                 /* per-TP dirty flag */
	struct list_head *tp_lru_nodes; /* per-TP node in tp_lru_list when TP is cached */
	struct list_head tp_lru_list;   /* TP-page level LRU list */
	struct dftl_cmt cmt;
	uint32_t num_tp;
	void *mapped;

	uint64_t *rmap; /* reverse mapptbl, assume it's stored in OOB */
	struct write_pointer wp;
	struct write_pointer gc_wp;
	struct line_mgmt lm;
	struct write_flow_control wfc;
};

void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size, void *mapped_addr,
			 uint32_t cpu_nr_dispatcher);

void conv_remove_namespace(struct nvmev_ns *ns);

bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req,
			   struct nvmev_result *ret);

void conv_ftl_get_debug_stats(u64 *hits, u64 *misses, u64 *loads, u64 *wbs);
void conv_ftl_reset_debug_stats(void);
void conv_ftl_set_debug_stats(u64 hits, u64 misses, u64 loads, u64 wbs);

#endif
