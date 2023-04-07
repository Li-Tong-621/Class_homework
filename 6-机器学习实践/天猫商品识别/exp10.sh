timer_start=`date "+%Y-%m-%d %H:%M:%S"`
export PYTHONIOENCONDING=utf-8
python exp1_data.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "数据处理耗时： $duration"
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python ./mmdetection/tools/train.py exp3.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "检测模型训练耗时： $duration"


python ./mmdetection/tools/test.py exp3.py ./expresult/det/latest.pth --format-only --options='jsonfile_prefix=./expresult/v_fs_cascade'
python exp5-2_coco.py
python exp5-3_filter.py


timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python exp6.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "识别模型训练耗时： $duration"


timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python exp7.py
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo eval $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "搜索和匹配耗时： $duration"

