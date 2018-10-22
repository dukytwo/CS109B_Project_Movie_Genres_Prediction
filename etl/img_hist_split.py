test_digit = 6

o_train = open('data/img_hist.tsv', 'w')
o_test = open('data/img_hist-test.tsv', 'w')
o = open('data/intermediate/img_hist.tsv')
header = o.readline()
header='tmdb_id	R_Count_0	R_Intensity_0	R_Count_1	R_Intensity_1	R_Count_2	R_Intensity_2	R_Count_3	R_Intensity_3	R_Count_4	R_Intensity_4	G_Count_0	G_Intensity_0	G_Count_1	G_Intensity_1	G_Count_2	G_Intensity_2	G_Count_3	G_Intensity_3	G_Count_4	G_Intensity_4	B_Count_0	B_Intensity_0	B_Count_1	B_Intensity_1	B_Count_2	B_Intensity_2	B_Count_3	B_Intensity_3	B_Count_4	B_Intensity_4	H_Count_0	H_Intensity_0	H_Count_1	H_Intensity_1	H_Count_2	H_Intensity_2	H_Count_3	H_Intensity_3	H_Count_4	H_Intensity_4	S_Count_0	S_Intensity_0	S_Count_1	S_Intensity_1	S_Count_2	S_Intensity_2	S_Count_3	S_Intensity_3	S_Count_4	S_Intensity_4	V_Count_0	V_Intensity_0	V_Count_1	V_Intensity_1	V_Count_2	V_Intensity_2	V_Count_3	V_Intensity_3	V_Count_4	V_Intensity_4'
o_train.write(header)
o_test.write(header)

for line in o:
    tmdb_id = int(line.split('\t')[0])
    tmdb_id_digit = tmdb_id %10
    if tmdb_id_digit < test_digit:
        o_train.write(line)
    elif tmdb_id_digit == test_digit:
        o_test.write(line)
        
