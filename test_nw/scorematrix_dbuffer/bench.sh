for i in 8 16 24 32 48 64 128
do
	for j in 1 2 3
	do
		./apitest_regular $i -10 | tee results_regular.txt
		./apitest_pinned $i -10 | tee results_pinned.txt
		./apitest_dualbuffer_pinned $i -10 | tee results_df_pinned.txt
	done
done
