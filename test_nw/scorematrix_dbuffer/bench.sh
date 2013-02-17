i=3
	for j in 1 2 3
	do
		./apitest_regular $i -10 | tee -a results_regular.txt
		./apitest_pinned $i -10 | tee -a results_pinned.txt
		./apitest_dualbuffer_pinned $i -10 | tee -a results_df_pinned.txt
	done


