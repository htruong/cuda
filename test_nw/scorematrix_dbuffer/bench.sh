i=3
	for j in 1 2 3
	do
		./apitest_regular $i -10 | tee -a results_regular.txt
		./apitest_pinned $i -10 | tee -a results_pinned.txt
		./apitest_dualbuffer_pinned $i -10 | tee -a results_db_pinned.txt
	done
cat results_db_pinned.txt | grep "CSV_ALL" > results_df_pinned.csv
cat results_pinned.txt | grep "CSV_ALL" > results_pinned.csv
cat results_regular.txt | grep "CSV_ALL" > results_regular.csv


