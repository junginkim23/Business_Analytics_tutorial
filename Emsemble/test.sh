test_mode=('vif' 'all')

for mode_1 in ${test_mode[@]}
do
    python main.py --mode_1 $mode_1
done