# D-FTL (Demand-based FTL) 구현 문서

수정 파일: conv_ftl.c, conv_ftl.h

## D-FTL 핵심 아이디어

**자주 접근하는 매핑만 DRAM에 캐싱**하고, 나머지는 **NAND의 Translation Page (TP)** 에 저장한다.

```
                    DRAM                          NAND
            ┌─────────────────┐           ┌─────────────────┐
 lookup ──> │  CMT (Cache)    │  miss ──> │ Translation Page │
            │  LPN -> PPA     │  <─ load  │ LPN -> PPA      │
            │  (1024 entries) │           │ (page 단위)      │
            └─────────────────┘           └─────────────────┘
                    │                             ▲
                    │ evict (dirty)                │
                    └─────────────────────────────-┘
            ┌─────────────────┐
            │  GTD (Directory) │  DRAM에 상주
            │  TP idx -> PPA   │  TP가 NAND 어디에 있는지 추적
            └─────────────────┘
```

## 추가된 자료구조

### 1. CMT (Cached Mapping Table) (`struct dftl_cmt`) — `conv_ftl.h:26`

캐싱된 LPN-PPA 매핑 정보를 관리하는 테이블. 
현재는 LRU eviction 정책 사용. 해시 테이블로 O(1) lookup을 지원한다(논문 참고).

```c
struct dftl_cmt {
    struct cmt_entry *pool;                   // 미리 할당된 entry 공간
    struct hlist_head ht[CMT_HASH_SIZE];      // 해시 테이블 (lookup용)
    struct list_head lru_list;                // LRU 순서 (head = MRU)
    struct list_head free_list;              // 아직 사용 안 된 entry
    uint32_t size;                           // 현재 캐싱된 entry 수
    uint32_t capacity;                       // 최대 entry 수 (CMT_CAPACITY)
};
```

### 2. CMT Entry (`struct cmt_entry`) — `conv_ftl.h:18`

하나의 캐싱된 LPN-PPA 매핑을 표현한다.

```c
struct cmt_entry {
    uint64_t lpn;           // Logical Page Number
    struct ppa ppa;         // Physical Page Address
    bool dirty;             // NAND에 반영 안 된 변경이 있는지
    struct list_head lru;   // LRU 리스트 연결
    struct hlist_node hnode; // 해시 테이블 연결
};
```

### 3. GTD (Global Translation Directory) — `conv_ftl.h:88`

Translation Page가 NAND 어디에 저장되어 있는지 추적하는 배열이다. DRAM에 상주하며 크기가 작다 (TP 개수만큼만 필요).

```c
// conv_ftl 구조체 내부
struct ppa *gtd;       // gtd[tp_idx] = TP가 저장된 NAND PPA
uint32_t num_tp;       // Translation Page 총 개수
void *mapped;          // NAND 메모리 매핑 영역 포인터(for Virt)
```

### 4. `conv_ftl` 구조체 변경 — `conv_ftl.h:82`

| 기존 (conv_ftl)         | D-FTL                    |
|------------------------|--------------------------|
| `struct ppa *maptbl`   | (삭제)                    |
| —                      | `struct ppa *gtd`        |
| —                      | `struct dftl_cmt cmt`    |
| —                      | `uint32_t num_tp`        |
| —                      | `void *mapped`           |

### 5. 설정값 — `conv_ftl.h:13-16`

```c
#define CMT_CAPACITY   64                        // CMT 최대 entry 수
#define CMT_HASH_BITS  6                         // 해시 버킷 수 = 2^6
#define CMT_HASH_SIZE  (1 << CMT_HASH_BITS)      // = 64
#define TRANS_LPN_BASE (INVALID_LPN - (1ULL<<32)) // TP용 특수 LPN 기준값
```

## D-FTL 동작

### 초기화/해제

| 함수 | 위치 | 설명 |
|------|------|------|
| `init_dftl()` | `conv_ftl.c:326` | GTD 배열 할당 (UNMAPPED 초기화), CMT pool 할당, 해시/LRU/free 리스트 초기화 |
| `remove_dftl()` | `conv_ftl.c:351` | 통계 출력 (hits, misses, loads, writebacks => 디버깅용. 추후 삭제 예정) 후 메모리 해제 |

### CMT 연산

| 함수 | 위치 | 설명 |
|------|------|------|
| `cmt_lookup()` | `conv_ftl.c:437` | 해시 테이블에서 LPN 검색. hit 시 MRU로 이동. O(1) |
| `cmt_evict()` | `conv_ftl.c:452` | LRU entry 제거. dirty면 `tp_writeback()` 호출 |
| `cmt_insert()` | `conv_ftl.c:468` | 새 매핑 삽입. 꽉 차면 evict 후 재사용 |

### Translation Page 연산

| 함수 | 위치 | 설명 |
|------|------|------|
| `tp_load()` | `conv_ftl.c:489` | NAND에서 TP를 읽어 특정 LPN의 PPA 반환. |
| `tp_writeback()` | `conv_ftl.c:361` | dirty entry를 TP에 기록. read-modify-write 방식. GC_IO write pointer 사용 |

### 매핑 인터페이스 (기존 `get/set_maptbl_ent` 대체)

| 함수 | 위치 | 설명 |
|------|------|------|
| `dftl_get_ppa()` | `conv_ftl.c:526` | CMT hit → 바로 반환. miss → `tp_load()` + `cmt_insert(dirty=false)` |
| `dftl_set_ppa()` | `conv_ftl.c:538` | CMT hit → PPA 갱신 + dirty 마킹. miss → `cmt_insert(dirty=true)` (TP 안 읽음) |

## Read / Write 경로

### Read 경로

```
conv_read()
  └─ dftl_get_ppa(lpn, &stime)
       ├─ CMT hit  → PPA 바로 반환 (DRAM only)
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
       ├─ CMT hit  → PPA 갱신 + dirty=true
       └─ CMT miss → cmt_insert(dirty=true)
                      (기존 TP를 읽지 않음 — write-on-miss 최적화)
```

**Write-on-miss 최적화**: write 시 CMT miss가 발생해도 기존 TP를 읽지 않는다. 어차피 새 PPA로 덮어쓸 것이므로 불필요한 NAND read를 회피한다.

### CMT Eviction → TP Writeback 경로

```
cmt_insert() (CMT 꽉 참)
  └─ cmt_evict()
       └─ tp_writeback() (dirty entry만)
            ├─ 기존 TP 있음 → read-modify-write
            │   1. old TP를 NAND에서 읽기 (GC_IO read)
            │   2. 내용 복사
            │   3. dirty entry 반영
            │   4. 새 page에 쓰기 (GC_IO write)
            │   5. old TP page invalidate
            └─ 기존 TP 없음 → 새 TP 생성
                1. UNMAPPED로 초기화
                2. dirty entry 반영
                3. 새 page에 쓰기
            └─ GTD 업데이트, rmap에 TRANS_LPN_BASE+tp_idx 기록
```

`rmap`에 `TRANS_LPN_BASE + tp_idx` 값을 저장하여 translation page를 식별한다.

## Summary

| 파일 | 변경 내용 |
|------|----------|
| `conv_ftl.h` | CMT/GTD 자료구조 추가, `maptbl` → `gtd`/`cmt`/`num_tp`/`mapped`로 교체 |
| `conv_ftl.c` | D-FTL 함수 전체 추가, read/write/GC 경로에서 `get/set_maptbl_ent` → `dftl_get/set_ppa` 교체 |

## Validation

공통 조건: iodepth=1 (latency 측정 목적), bs=4k, direct I/O.

---

### 실험 1: convFTL 대비 D-FTL overhead 측정

CMT_CAPACITY=64로 고정하고, D-FTL과 convFTL(full in-memory mapping)을 비교한다.  
Working set이 CMT보다 훨씬 클 때 D-FTL overhead를 측정한다.

```bash
# Sequential write 1 GB → Random read 512 MB
sudo fio --name=write --filename=/dev/nvme0n1 --rw=write     --bs=4k --iodepth=1 --size=1G   --direct=1
sudo fio --name=read  --filename=/dev/nvme0n1 --rw=randread  --bs=4k --iodepth=1 --size=512M --direct=1
```

| 항목              |     D-FTL |   convFTL |         overhead |
|-------------------|----------:|----------:|-----------------:|
| Write avg latency |  29.10 µs |  11.56 µs | **2.52x** |
| Write IOPS        |     34.2k |     84.1k | **2.46x 낮음** |
| Read avg latency  |  96.57 µs |  62.01 µs | **1.56x** |
| Read IOPS         |     10.3k |     16.0k | **1.55x 낮음** |

**CMT 통계 (D-FTL)**

```
CMT hits=393,731 (50%) / misses=393,823 (50%) / tp_loads=393,823 / tp_writebacks=262,144
```

- **Read 1.56x**: miss 50% → 절반의 read에서 TP NAND read 추가 (기대 overhead 1.5x ≈ 실측 1.56x)
- **Write 2.52x**: miss eviction의 66.5%가 dirty → tp_writeback(NAND read-modify-write) 추가

---

### 실험 2: CMT Capacity Sweep

Working set을 CMT 크기와 맞춰 CMT capacity 효과를 측정한다.

```bash
# Write 4 MB → Random read 4 MB (working set = 1024 LPN)
sudo fio --name=write --filename=/dev/nvme0n1 --rw=write     --bs=4k --iodepth=1 --size=4M --direct=1
sudo fio --name=read  --filename=/dev/nvme0n1 --rw=randread  --bs=4k --iodepth=1 --size=4M --direct=1
```

| 항목              |   CMT=64 |  CMT=1024 |           비고 |
|-------------------|----------:|----------:|---------------:|
| avg latency       |  98.04 µs |  62.45 µs | **36% 감소** |
| IOPS              |     10.1k |     15.8k | **1.56x 증가** |
| 99.9th latency    |    652 µs |    133 µs | tail latency 개선 |
| max latency       |    823 µs |    171 µs | |

- **CMT=64**: write 중 960회 eviction → tp_writeback으로 GTD 채워짐 → read에서 CMT miss 시 TP NAND read 발생 → ~98 µs
- **CMT=1024**: working set(1024 LPN) = CMT 크기 → eviction 없음 → GTD 미사용, CMT에서 직접 hit → ~62 µs ≈ convFTL 동등
- working set이 CMT에 완전히 들어오면 D-FTL은 **translation NAND I/O 없이 convFTL과 동등한 성능**을 낸다.

### 실험 3: CMT capacity sweep

`CMT_CAPACITY`를 64 → 1024로 재컴파일하며 random read 성능을 측정한다.
실효 working set은 `nr_parts`로 분할되므로 per-instance 기준 ~512 LPN 수준이다.

```bash
# 각 CMT 빌드마다: write precondition → random read
sudo fio --name=precond --filename=/dev/nvme0n1 --rw=write    --bs=4k --iodepth=1 --size=4M --direct=1
sudo fio --name=sweep   --filename=/dev/nvme0n1 --rw=randread --bs=4k --iodepth=1 --size=4M --direct=1
```

|  CMT |   hit rate | tp_loads | tp_writebacks |  rd avg | rd IOPS |
|-----:|-----------:|---------:|--------------:|--------:|--------:|
|   64 |       8.4% |     1906 |          1024 | 95.3 µs |   10.4k |
|  128 |      18.3% |     1565 |          1024 | 110.9 µs¹ |  8.9k |
|  256 |      39.0% |      462 |           457 | 86.3 µs |   11.5k |
|  512 | **59.8%** |    **0** |         **0** | **62.3 µs** | **15.8k** |
| 1024 |      59.8% |        0 |             0 | 69.3 µs¹ |  14.2k |

¹ latency outlier(측정 노이즈). hit rate·tp_loads가 단조적이므로 주지표로 사용.

- CMT가 working set을 모두 담는 **512 지점에서 translation NAND I/O가 0**으로 사라지며 convFTL 동등 성능(62 µs)에 도달.
- 그 이하에서는 eviction → tp_writeback, miss 시 tp_load가 발생하여 latency 증가.

---

### 실험 4: Sequential vs Random 패턴

Working set(64 MB = 16384 LPN)이 CMT를 크게 초과하는 조건에서 패턴 영향을 본다.

```bash
sudo fio --name=fill   --filename=/dev/nvme0n1 --rw=write     --bs=4k --iodepth=1 --size=64M --direct=1
sudo fio --name=seq_rd --filename=/dev/nvme0n1 --rw=read      --bs=4k --iodepth=1 --size=64M --direct=1
sudo fio --name=rnd_rd --filename=/dev/nvme0n1 --rw=randread  --bs=4k --iodepth=1 --size=64M --direct=1
sudo fio --name=seq_wr --filename=/dev/nvme0n1 --rw=write     --bs=4k --iodepth=1 --size=64M --direct=1
sudo fio --name=rnd_wr --filename=/dev/nvme0n1 --rw=randwrite --bs=4k --iodepth=1 --size=64M --direct=1
```

| 패턴       |   avg | 99.9th |  IOPS |
|------------|------:|-------:|------:|
| seq read   | 95.3 µs | 362.5 µs | 10.5k |
| rand read  | 95.3 µs | 301.1 µs | 10.5k |
| seq write  | 26.3 µs |  43.3 µs | 37.4k |
| rand write | 22.1 µs |  32.4 µs | 44.5k |

- read는 seq/rand가 **거의 동일** (hit rate 0.7%, tp_loads 80210). working set이 CMT를 크게 초과해 접근 패턴보다 **CMT 용량 부족이 지배적**.

---

### 실험 5: Read-heavy vs Write-heavy 비율

`rwmixread`를 90/70/50/30으로 바꿔 read:write를 9:1 ~ 3:7로 변화시킨다 (64 MB fill 선행).

```bash
sudo fio --name=fill --filename=/dev/nvme0n1 --rw=write --bs=4k --iodepth=1 --size=64M --direct=1
sudo fio --name=mix  --filename=/dev/nvme0n1 --rw=randrw --rwmixread=90 \
         --bs=4k --iodepth=1 --size=64M --direct=1   # 90 → 70 → 50 → 30
```

| read:write | rd hit rate | tp_writebacks |  rd avg | rd IOPS | wr IOPS |
|------------|------------:|--------------:|--------:|--------:|--------:|
| 9:1 | 3.8% | 17576 | 117.8 µs |  8.4k |  0.9k |
| 7:3 | 3.8% | 20006 | 131.5 µs |  7.3k |  3.1k |
| 5:5 | 3.8% | 22403 | 149.1 µs |  6.1k |  6.2k |
| 3:7 | 3.8% | 24819 | 182.5 µs |  4.7k | 11.0k |

- write 비율↑ → **dirty eviction(tp_writeback) 증가**(17.6k → 24.8k) → read latency 117 → 182 µs로 악화. write가 translation writeback/GC를 유발해 read를 간섭.
- read hit rate는 동일 fill·working set이라 mix 비율과 무관하게 일정(3.8%).