# D-FTL (Demand-based FTL) 구현 문서

수정 파일: conv_ftl.c, conv_ftl.h

## D-FTL 핵심 아이디어

**자주 접근하는 매핑만 DRAM에 캐싱**하고, 나머지는 **NAND의 Translation Page (TP)** 에 저장한다.

```
                    DRAM                          NAND
            ┌─────────────────┐           ┌─────────────────┐
 lookup ──> │  CMT (Cache)    │  miss ──> │ Translation Page │
            │  LPN -> PPA     │  <─ load  │ LPN -> PPA      │
            │  (TPnode 단위)  │           │ (page 단위)      │
            └─────────────────┘           └─────────────────┘
                    │                             ▲
                    │ evict (dirty batch)          │
                    └─────────────────────────────-┘
            ┌─────────────────┐
            │  GTD (Directory) │  DRAM에 상주
            │  TP idx -> PPA   │  TP가 NAND 어디에 있는지 추적
            └─────────────────┘
```

CMT는 **TPnode 단위**로 관리한다. 같은 translation page에 속한 캐싱 entry들을 한 TPnode 아래 묶고, **LRU도 TPnode 단위**로 관리한다. evict이 발생하면 LRU tail TPnode 하나를 통째로 비우며, 그 안의 dirty entry들은 **한 번의 read-modify-write**로 함께 NAND에 반영된다 (batch flush).

## 추가된 자료구조

### 1. CMT (Cached Mapping Table) (`struct dftl_cmt`) — `conv_ftl.h:36`

entry 풀과 노드 풀을 모두 관리한다. **LRU는 entry가 아니라 TPnode 단위**로 유지한다.

```c
struct dftl_cmt {
    struct cmt_entry *entry_pool;                   // entry 미리 할당
    struct tp_node   *node_pool;                    // TPnode 미리 할당
    struct hlist_head entry_ht[CMT_ENTRY_HASH_SIZE]; // entry 해시 (key: lpn)
    struct hlist_head node_ht [CMT_NODE_HASH_SIZE];  // node 해시 (key: tp_idx)
    struct list_head node_lru;                       // TPnode LRU (head=MRU)
    struct list_head entry_free_list;
    struct list_head node_free_list;
    uint32_t entry_size, entry_capacity;
    uint32_t node_size,  node_capacity;
};
```

### 2. TPnode (`struct tp_node`) — `conv_ftl.h:27`

하나의 translation page에 속한 캐싱 entry들을 묶는 그룹 노드.

```c
struct tp_node {
    uint32_t tp_idx;
    uint32_t entry_count;
    uint32_t dirty_count;            // 빠른 "evict 시 writeback 필요?" 판단
    struct list_head entries;        // cmt_entry->sibling
    struct list_head lru;            // dftl_cmt->node_lru
    struct hlist_node hnode;         // dftl_cmt->node_ht
};
```

### 3. CMT Entry (`struct cmt_entry`) — `conv_ftl.h:18`

하나의 캐싱된 LPN-PPA 매핑. 항상 어떤 TPnode에 매달려 있다.

```c
struct cmt_entry {
    uint64_t lpn;
    struct ppa ppa;
    bool dirty;
    struct tp_node *parent;          // 소속 TPnode
    struct list_head sibling;        // parent->entries
    struct hlist_node hnode;         // entry 해시 (key: lpn)
};
```

### 4. GTD (Global Translation Directory) — `conv_ftl.h:91`

Translation Page가 NAND 어디에 저장되어 있는지 추적하는 배열이다. DRAM에 상주하며 크기가 작다 (TP 개수만큼만 필요).

```c
// conv_ftl 구조체 내부
struct ppa *gtd;       // gtd[tp_idx] = TP가 저장된 NAND PPA
uint32_t num_tp;       // Translation Page 총 개수
void *mapped;          // NAND 메모리 매핑 영역 포인터(for Virt)
```

### 5. `conv_ftl` 구조체 변경 — `conv_ftl.h:85`

| 기존 (conv_ftl)         | D-FTL                    |
|------------------------|--------------------------|
| `struct ppa *maptbl`   | (삭제)                    |
| —                      | `struct ppa *gtd`        |
| —                      | `struct dftl_cmt cmt`    |
| —                      | `uint32_t num_tp`        |
| —                      | `void *mapped`           |

### 6. 설정값 — `conv_ftl.h:13-17`

```c
#define CMT_CAPACITY        64                        // entry 최대 수 (노드 풀 크기도 동일)
#define CMT_ENTRY_HASH_BITS 6                         // entry 해시 버킷
#define CMT_NODE_HASH_BITS  6                         // node 해시 버킷
#define TRANS_LPN_BASE      (INVALID_LPN - (1ULL<<32)) // TP용 특수 LPN 기준값
```

노드 풀은 최악(모든 entry가 서로 다른 TP)에 맞춰 entry 풀과 동일 크기로 잡는다.

## D-FTL 동작

### 초기화/해제

| 함수 | 위치 | 설명 |
|------|------|------|
| `init_dftl()` | `conv_ftl.c:333` | GTD 배열 할당 (UNMAPPED 초기화), entry/node pool 할당, 두 해시테이블/LRU/free 리스트 초기화 |
| `remove_dftl()` | `conv_ftl.c:363` | 통계 출력 (hits, misses, loads, writebacks, node_evicts, avg_batch) 후 메모리 해제 |

`avg_batch`는 writeback 한 번당 평균 함께 flush된 dirty entry 수. 1.0보다 크면 TPnode batching이 동작 중이라는 신호.

### CMT 연산

| 함수 | 위치 | 설명 |
|------|------|------|
| `cmt_lookup()` | `conv_ftl.c:473` | entry 해시에서 lpn 검색. hit 시 **parent TPnode**를 node_lru MRU로 이동. O(1) |
| `node_lookup()` | `conv_ftl.c:380` | node 해시에서 tp_idx 검색. O(1) |
| `cmt_evict_node()` | `conv_ftl.c:489` | LRU tail TPnode를 통째로 evict. `dirty_count > 0`이면 `tp_writeback_node()` 1회 호출. 노드의 모든 entry를 free, 노드도 free |
| `cmt_insert()` | `conv_ftl.c:514` | tp_idx 계산 → `node_lookup` → 없으면 신규 노드 할당. entry 풀 부족 시 evict (retry로 자기 노드가 evict되는 degenerate 케이스 대응). entry를 노드에 매달고 dirty면 `node->dirty_count++` |

### Translation Page 연산

| 함수 | 위치 | 설명 |
|------|------|------|
| `tp_load()` | `conv_ftl.c:565` | NAND에서 TP를 읽어 특정 LPN의 PPA 반환 |
| `tp_writeback_node()` | `conv_ftl.c:391` | TPnode의 **모든 dirty entry를 한 번의 RMW로 함께** TP에 기록. GC_IO write pointer 사용. 종료 후 각 entry dirty=false, node->dirty_count=0 |

### 매핑 인터페이스 (기존 `get/set_maptbl_ent` 대체)

| 함수 | 위치 | 설명 |
|------|------|------|
| `dftl_get_ppa()` | `conv_ftl.c:603` | CMT hit → 바로 반환. miss → `tp_load()` + `cmt_insert(dirty=false)` |
| `dftl_set_ppa()` | `conv_ftl.c:615` | CMT hit → PPA 갱신, clean→dirty 전환 시 `parent->dirty_count++`. miss → `cmt_insert(dirty=true)` (TP 안 읽음) |

## Read / Write 경로

### Read 경로

```
conv_read()
  └─ dftl_get_ppa(lpn, &stime)
       ├─ CMT hit  → PPA 바로 반환 (DRAM only) + parent TPnode를 MRU로
       └─ CMT miss → tp_load()로 NAND에서 TP 읽기
                      → cmt_insert(dirty=false)로 캐싱
                      → PPA 반환
```

miss 시 NAND read latency가 추가로 발생한다.

### Write 경로

```
conv_write()
  ├─ dftl_get_ppa(lpn)  // 기존 매핑 확인 (overwrite 시 old page invalidate용)
  ├─ get_new_page()      // 새 physical page 할당
  └─ dftl_set_ppa(lpn, new_ppa, &stime)
       ├─ CMT hit  → PPA 갱신 + dirty=true (clean→dirty면 parent->dirty_count++)
       └─ CMT miss → cmt_insert(dirty=true)
                      (기존 TP를 읽지 않음 — write-on-miss 최적화)
```

**Write-on-miss 최적화**: write 시 CMT miss가 발생해도 기존 TP를 읽지 않는다. 어차피 새 PPA로 덮어쓸 것이므로 불필요한 NAND read를 회피한다.

### CMT Eviction → TP Writeback 경로 (batch flush)

```
cmt_insert() (entry 풀 가득)
  └─ cmt_evict_node() (LRU tail TPnode)
       ├─ dirty_count > 0 이면 tp_writeback_node() 1회
       │    ├─ 기존 TP 있음 → read-modify-write
       │    │   1. old TP를 NAND에서 읽기 (GC_IO read)
       │    │   2. 내용 복사
       │    │   3. 노드의 모든 dirty entry를 buffer에 한꺼번에 patch  ★
       │    │   4. 새 page에 쓰기 (GC_IO write)
       │    │   5. old TP page invalidate
       │    └─ 기존 TP 없음 → 새 TP 생성
       │         1. UNMAPPED로 초기화
       │         2. 노드의 모든 dirty entry를 buffer에 한꺼번에 patch  ★
       │         3. 새 page에 쓰기
       │    └─ GTD 업데이트, rmap에 TRANS_LPN_BASE+tp_idx 기록
       └─ 노드의 모든 entry를 entry_free_list로 반환, 노드도 node_free_list로 반환
```

★ 부분이 핵심 변경. 기존 (entry-LRU) 설계에선 같은 TP에 속한 dirty entry가 여러 번 evict될 때마다 매번 별도의 RMW가 발생했는데, TPnode-LRU에선 한 번에 묶어서 처리하므로 **TP에 대한 NAND write 횟수가 dirty entry 수만큼이 아니라 TPnode evict 횟수만큼**으로 줄어든다.

`rmap`에 `TRANS_LPN_BASE + tp_idx` 값을 저장하여 translation page를 식별한다.

### Degenerate 케이스: 단일 TPnode가 entry 풀을 다 차지하는 경우

워크로드가 한 TP 영역(같은 tp_idx)만 반복 접근하면 노드는 1개, entry는 entry_capacity만큼 채워질 수 있다. 이 상태에서 같은 TP의 새 lpn을 insert하면 LRU tail이 자기 자신이 되어 자기 노드가 evict된다. `cmt_insert()`는 `retry` 라벨로 다시 lookup하여 새 노드를 만든다 (이전 entry들은 batch writeback으로 NAND에 안전하게 반영된 상태).

## Summary

| 파일 | 변경 내용 |
|------|----------|
| `conv_ftl.h` | TPnode 자료구조 추가, `cmt_entry`에 `parent`/`sibling` 필드, `dftl_cmt`를 entry/node 두 풀로 확장 |
| `conv_ftl.c` | LRU 단위를 TPnode로 전환, `cmt_evict_node` 및 `tp_writeback_node` (batch RMW), `node_lookup` 추가, `cmt_insert` 재작성 |

## 통계 출력 예시 (`remove_dftl`)

```
CMT hits=12345 misses=678 tp_loads=678 tp_writebacks=12 node_evicts=45 avg_batch=3.75
```

- `avg_batch > 1.0` → TPnode batching이 실제로 동작 (한 writeback에 평균 N개 dirty entry 묶임)
- `node_evicts >= tp_writebacks` → clean 노드 evict는 writeback 없이 빠르게 회수
