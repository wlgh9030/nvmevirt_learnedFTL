sudo insmod ./nvmev.ko memmap_start=4G memmap_size=32G cpus=14,15

sudo mkfs.ext4 -F /dev/nvme2n1

sudo mount /dev/nvme2n1 /mnt/virt



  # 1) dmesg 비우고 시작
 # sudo dmesg -C

  # 2) GC를 유발할 만큼 쓰기 (디바이스 크기보다 크게, 순차가 학습에 유리)
  #    /dev/nvmeXn1 을 실제 디바이스로 바꾸세요. 마운트했으면 파일 기반으로:
#  sudo fio --name=seqwrite --filename=/mnt/virt/testfile \
  #    --rw=write --bs=128k --size=2G --ioengine=io_uring --iodepth=32 --direct=1

  # 3) 같은 영역을 다시 써서 GC 확실히 유발 (overwrite)
 # sudo fio --name=overwrite --filename=/mnt/virt/testfile \
   #   --rw=randwrite --bs=4k --size=2G --ioengine=io_uring --iodepth=32 --direct=1

  # 4) 읽기 (CMT를 넘어서는 범위를 읽어야 miss→예측 경로가 돈다)
 # sudo fio --name=read --filename=/mnt/virt/testfile \
    #  --rw=read --bs=4k --size=2G --ioengine=io_uring --iodepth=32 --direct=1

