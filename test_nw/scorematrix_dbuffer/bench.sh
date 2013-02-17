for i in 8 16 24 32 48 64 128
do
 	echo -e "----- TESTING WITH $i PAIRS ----" | tee -a results_regular.txt
 	echo -e "----- TESTING WITH $i PAIRS ----" | tee -a results_pinned.txt
 	echo -e "----- TESTING WITH $i PAIRS ----" | tee -a results_df_pinned.txt
	for j in 1 2 3
	do
		./apitest_regular $i -10 | tee -a results_regular.txt
		./apitest_pinned $i -10 | tee -a results_pinned.txt
		./apitest_dualbuffer_pinned $i -10 | tee -a results_df_pinned.txt
	done
done
