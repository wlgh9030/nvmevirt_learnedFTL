#!/bin/bash
# D-FTL / learnedFTL 테스트 러너
#   - 정합성(버그) 검증: fio --verify 로 "쓴 데이터 == 읽은 데이터" 확인
#   - learnedFTL 효과 측정: 순차 쓰기로 모델 학습 유발 후 랜덤 읽기로 예측 경로 가동
#
# 주의: raw 디바이스에 직접 쓰므로 $DEV 의 내용은 모두 지워집니다 (테스트 전용).
# 사용법: ./test_ftl.sh [bug-seq | bug-rand | bug-rw | learned | paper-randread | paper-seqread | paper-read | sweep | all]
set -u

DEV=/dev/nvme3n1
KO=./nvmev.ko
MEMMAP="memmap_start=4G memmap_size=28G cpus=14,15 gc_cpus=13"   # gc_cpus = background GC thread (isolcpus 중 하나)
SIZE=3G          # 디바이스보다 작게 (OP 제외 ~3.7G). GC 유발은 loops 로.
COMMON="--filename=$DEV --ioengine=io_uring --direct=1 --group_reporting"
PAPER_THREADS=64
PAPER_BS=4k
PAPER_WARMUP_LOOPS=6
PAPER_WARMUP_COMMON="--filename=$DEV --ioengine=psync --direct=1 --group_reporting --bs=$PAPER_BS --iodepth=1"
PAPER_READ_COMMON="--filename=$DEV --ioengine=psync --direct=1 --group_reporting --thread --numjobs=$PAPER_THREADS --bs=$PAPER_BS --iodepth=1"

load()   { sudo dmesg -C; sudo insmod $KO $MEMMAP || exit 1; sleep 1; }
stats() {
  sudo dmesg | grep -E "NVMeVirt: (CMT|LR model|GC diag|WA)" | tail -4
  sudo dmesg | awk '
    /NVMeVirt: CMT / {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^hits=/) {
          split($i, a, "=");
          cmt_hits = a[2] + 0;
        } else if ($i ~ /^misses=/) {
          split($i, a, "=");
          cmt_misses = a[2] + 0;
        }
      }
    }
    /NVMeVirt: LR model/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^uses=/) {
          split($i, a, "=");
          model_uses = a[2] + 0;
        } else if ($i ~ /^hits=/) {
          split($i, a, "=");
          model_hits = a[2] + 0;
        }
      }
    }
    END {
      cmt_access = cmt_hits + cmt_misses;
      if (cmt_access > 0) {
        printf "Fig14 ratios: CMT hit %.2f%%, Model hit %.2f%% of CMT misses, Combined %.2f%% of reads\n",
          cmt_hits * 100.0 / cmt_access,
          cmt_misses ? model_hits * 100.0 / cmt_misses : 0,
          (cmt_hits + model_hits) * 100.0 / cmt_access;
      }
      if (model_uses > 0) {
        printf "LR verify ratio: %.2f%% (hits %.0f / uses %.0f)\n",
          model_hits * 100.0 / model_uses, model_hits, model_uses;
      }
    }'
}
unload() { sudo rmmod nvmev; echo "----- stats -----"; stats; echo; }

paper_range_bytes() {
  local devb=$1
  local range

  range=$((devb / PAPER_THREADS / 4096 * 4096))
  if [ "$range" -le 0 ]; then
    echo "ERROR: $DEV is too small for PAPER_THREADS=$PAPER_THREADS" >&2
    exit 1
  fi

  echo "$range"
}

reset_read_cmt_stat() {
  if [ -e /proc/nvmev/cmt_stat ]; then
    echo 1 | sudo tee /proc/nvmev/cmt_stat >/dev/null
  fi
}

show_read_cmt_stat() {
  echo "----- read-phase CMT stat -----"
  if [ -e /proc/nvmev/cmt_stat ]; then
    sudo awk '{
      if ($2 > 0)
        printf "%s (CMT hit ratio %.2f%%)\n", $0, ($4 * 100.0 / $2);
      else
        printf "%s (CMT hit ratio n/a)\n", $0;
    }' /proc/nvmev/cmt_stat
  else
    echo "WARN: /proc/nvmev/cmt_stat not found"
  fi
}

# ---------------------------------------------------------------------------
# 1) 정합성 — 순차 write 후 자동 verify read (가장 기본적인 매핑 깨짐 탐지)
bug_seq() {
  echo "### [bug-seq] sequential write + verify read"
  load
  sudo fio --name=seq_integ $COMMON --rw=write --bs=128k --size=$SIZE \
      --iodepth=32 --verify=crc32c --do_verify=1 --verify_fatal=1
  r=$?
  [ $r -eq 0 ] && echo "PASS: 데이터 정합성 OK" || echo "FAIL($r): 데이터 깨짐 발견!"
  unload
}

# 2) 정합성 — 랜덤 write 후 자동 verify read (랜덤 매핑 경로 검증)
bug_rand() {
  echo "### [bug-rand] random write + verify read"
  load
  sudo fio --name=rand_integ $COMMON --rw=randwrite --bs=4k --size=$SIZE \
      --iodepth=16 --serialize_overlap=1 \
      --verify=crc32c --do_verify=1 --verify_fatal=1
  r=$?
  [ $r -eq 0 ] && echo "PASS: 데이터 정합성 OK" || echo "FAIL($r): 데이터 깨짐 발견!"
  unload
}

# 3) 정합성 — rw 인터리브 (버그가 가장 잘 터지는 경로)
#    verify_backlog: N블록 쓸 때마다 그 블록들을 즉시 되읽어 검증 → 쓰기 직후
#    stale 매핑(model_predict 오예측, tp_map/CMT 불일치)을 바로 노출.
#    serialize_overlap: 같은 LBA 동시 op 직렬화 → 검증 false positive 방지.
bug_rw() {
  echo "### [bug-rw] interleaved write/verify (read-after-write 매핑 검증)"
  load
  sudo fio --name=rw_integ $COMMON --rw=randwrite --bs=4k --size=$SIZE \
      --iodepth=16 --serialize_overlap=1 --loops=2 \
      --verify=crc32c --verify_backlog=1024 --verify_backlog_batch=1024 \
      --do_verify=1 --verify_fatal=1
  r=$?
  [ $r -eq 0 ] && echo "PASS: 데이터 정합성 OK" || echo "FAIL($r): 데이터 깨짐 발견!"
  unload
}

# ---------------------------------------------------------------------------
# 4) learnedFTL 효과 — 순차 fill 로 라인을 연속 lpn 으로 채운 뒤, "가벼운" 랜덤
#    overwrite 로 victim 라인을 부분 무효화한다. 그래야 GC 가 valid 한 연속 런을
#    재배치하고(=학습 샘플 多), 그 샘플들이 같은 tp_idx 에 밀집해 임계값을 넘긴다.
#    (순수 순차 overwrite 는 라인이 통째 무효화돼 재배치할 valid 가 0 → 학습 0.)
#    이후 랜덤 read 로 CMT(64) miss 폭증 → model_predict 대량 가동.
learned() {
  echo "### [learned] seq fill -> light random overwrite -> random read"
  load
  local DEVB SZ OW
  DEVB=$(sudo blockdev --getsize64 $DEV)
  SZ=$((DEVB * 90 / 100))   # GC 압박은 유지하되 OP 여유를 조금 더 둠
  OW=$((DEVB * 20 / 100))   # free 공간보다 크게 overwrite → GC 유발, victim 은 다수 valid 유지
  echo "-- dev=$DEVB  fill=$SZ  overwrite=$OW (bytes)"
  echo "-- phase1: sequential fill"
  sudo fio --name=seq_fill $COMMON --rw=write --bs=128k --size=$SZ --iodepth=32
  echo "-- phase2: light random overwrite (부분 무효화 → valid 재배치 유발)"
  sudo fio --name=light_ow $COMMON --rw=randwrite --bs=4k --size=$SZ \
      --io_size=$OW --iodepth=16 --serialize_overlap=1
  echo "-- phase3: random read (CMT miss -> 예측 경로)"
  sudo fio --name=rand_read $COMMON --rw=randread --bs=4k --size=$SZ --iodepth=32 
  echo "-- 결과 (LR hits / GC diag 확인):"
  unload
}

# ---------------------------------------------------------------------------
# 5) Fig. 14(b) 논문 조건: stable-state warm-up 후 4KB / psync / 64 threads read.
#    - RandRead: random write로 전체 SSD를 약 6회 warm-up 후 randread
#    - SeqRead : sequential write로 전체 SSD를 약 6회 warm-up 후 read
#    warm-up은 단일 stream으로 전체 SSD를 6회 쓴다. 특히 sequential warm-up을
#    64개 disjoint stream으로 만들면 SI의 contiguous run이 깨져 model coverage가
#    낮아진다. read phase만 64 threads를 사용한다.
paper_read_workload() {
  local name=$1
  local warm_rw=$2
  local read_rw=$3
  local DEVB RANGE COVER r

  echo "### [paper-$name] Fig. 14(b): $warm_rw warm-up -> $read_rw"
  load

  DEVB=$(sudo blockdev --getsize64 $DEV)
  RANGE=$(paper_range_bytes "$DEVB")
  COVER=$((RANGE * PAPER_THREADS))
  echo "-- dev=$DEVB bytes, covered=$COVER bytes, per_thread_range=$RANGE bytes"
  echo "-- warm-up: rw=$warm_rw bs=$PAPER_BS ioengine=psync threads=1 loops=$PAPER_WARMUP_LOOPS"
  sudo fio --name="${name}_warmup" $PAPER_WARMUP_COMMON --rw="$warm_rw" \
      --size="$COVER" --loops="$PAPER_WARMUP_LOOPS"
  r=$?
  if [ $r -ne 0 ]; then
    echo "FAIL($r): warm-up fio failed"
    unload
    return $r
  fi

  echo "-- reset read-path hit counters before read benchmark"
  reset_read_cmt_stat

  echo "-- read benchmark: rw=$read_rw bs=$PAPER_BS ioengine=psync threads=$PAPER_THREADS"
  sudo fio --name="${name}_read" $PAPER_READ_COMMON --rw="$read_rw" \
      --size="$RANGE" --offset_increment="$RANGE" --loops=1
  r=$?

  show_read_cmt_stat
  echo "-- unload log below prints LR model uses/hits for model hit check"
  unload
  return $r
}

paper_randread() { paper_read_workload "randread" "randwrite" "randread"; }
paper_seqread()  { paper_read_workload "seqread"  "write"     "read"; }
paper_read()     { paper_randread && paper_seqread; }

# ---------------------------------------------------------------------------
# 6) 성능 스윕 — bs x qd 그리드를 4개 워크로드로 측정해 collect.py 입력 포맷
#    (<dir>/<bs>_<qd>.json) 으로 저장한 뒤 collect.py 로 summary.csv + 그래프 생성.
#    전체 그리드는 7bs x 9qd x 4워크로드 = 252회 (회당 $SWEEP_RUNTIME 초) — 오래
#    걸리면 아래 리스트를 줄여서 사용.
SWEEP_DIR=./fio_results
SWEEP_BS_LIST="4k 8k 16k 32k 64k 128k 256k"
SWEEP_QD_LIST="1 2 4 8 16 32 64 128 256"
SWEEP_RUNTIME=5

sweep() {
  echo "### [sweep] bs x qd grid -> $SWEEP_DIR -> collect.py"
  load
  local wl rw bs qd dir r
  # write 워크로드를 먼저 돌려 read 워크로드가 매핑된 LPN 을 읽게 한다
  for wl in sequentialwrite randomwrite sequentialread randomread; do
    case $wl in
      sequentialwrite) rw=write ;;
      randomwrite)     rw=randwrite ;;
      sequentialread)  rw=read ;;
      randomread)      rw=randread ;;
    esac
    dir=$SWEEP_DIR/$wl
    mkdir -p "$dir"
    for bs in $SWEEP_BS_LIST; do
      for qd in $SWEEP_QD_LIST; do
        echo "-- $wl bs=$bs qd=$qd"
        sudo fio --name="${wl}_${bs}_${qd}" $COMMON --rw=$rw --bs=$bs \
            --iodepth=$qd --size=$SIZE --time_based --runtime=$SWEEP_RUNTIME \
            --output-format=json --output="$dir/${bs}_${qd}.json"
        r=$?
        [ $r -ne 0 ] && echo "WARN($r): fio failed at $wl bs=$bs qd=$qd"
      done
    done
  done
  unload
  echo "-- collect.py: summary.csv + graphs/"
  python3 "$(dirname "$0")/../collect.py" --base "$SWEEP_DIR"
}

case "${1:-all}" in
  bug-seq)        bug_seq ;;
  bug-rand)       bug_rand ;;
  bug-rw)         bug_rw ;;
  learned)        learned ;;
  paper-randread) paper_randread ;;
  paper-seqread)  paper_seqread ;;
  paper-read)     paper_read ;;
  sweep)          sweep ;;
  all)            bug_seq; bug_rand; bug_rw; learned ;;
  *) echo "usage: $0 [bug-seq|bug-rand|bug-rw|learned|paper-randread|paper-seqread|paper-read|sweep|all]"; exit 1 ;;
esac
