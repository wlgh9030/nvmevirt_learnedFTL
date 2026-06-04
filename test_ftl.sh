#!/bin/bash
# D-FTL / learnedFTL 테스트 러너
#   - 정합성(버그) 검증: fio --verify 로 "쓴 데이터 == 읽은 데이터" 확인
#   - learnedFTL 효과 측정: 순차 쓰기로 모델 학습 유발 후 랜덤 읽기로 예측 경로 가동
#
# 주의: raw 디바이스에 직접 쓰므로 $DEV 의 내용은 모두 지워집니다 (테스트 전용).
# 사용법: ./test_ftl.sh [bug-seq | bug-rand | bug-rw | learned | all]
set -u

DEV=/dev/nvme2n1
KO=./nvmev.ko
MEMMAP="memmap_start=4G memmap_size=4G cpus=14,15"
SIZE=3G          # 디바이스보다 작게 (OP 제외 ~3.7G). GC 유발은 loops 로.
COMMON="--filename=$DEV --ioengine=io_uring --direct=1 --group_reporting"

load()   { sudo dmesg -C; sudo insmod $KO $MEMMAP || exit 1; sleep 1; }
stats()  { sudo dmesg | grep -E "NVMeVirt: (CMT|LR model)" | tail -2; }
unload() { echo "----- stats -----"; stats; sudo rmmod nvmev; echo; }

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
  SZ=$((DEVB * 92 / 100))   # 거의 꽉 채워 GC 압박 ↑ (OP 여유는 조금)
  OW=$((DEVB * 25 / 100))   # free 공간보다 크게 overwrite → GC 확실 유발, victim 은 다수 valid 유지
  echo "-- dev=$DEVB  fill=$SZ  overwrite=$OW (bytes)"
  echo "-- phase1: sequential fill"
  sudo fio --name=seq_fill $COMMON --rw=write --bs=128k --size=$SZ --iodepth=32
  echo "-- phase2: light random overwrite (부분 무효화 → valid 재배치 유발)"
  sudo fio --name=light_ow $COMMON --rw=randwrite --bs=4k --size=$SZ \
      --io_size=$OW --iodepth=16 --serialize_overlap=1
  echo "-- phase3: random read (CMT miss -> 예측 경로)"
  sudo fio --name=rand_read $COMMON --rw=randread --bs=4k --size=$SZ --iodepth=32 
  echo "-- 결과 (LR hits / GC diag 확인):"
  stats
  sudo rmmod nvmev; echo
}

case "${1:-all}" in
  bug-seq)  bug_seq ;;
  bug-rand) bug_rand ;;
  bug-rw)   bug_rw ;;
  learned)  learned ;;
  all)      bug_seq; bug_rand; bug_rw; learned ;;
  *) echo "usage: $0 [bug-seq|bug-rand|bug-rw|learned|all]"; exit 1 ;;
esac
