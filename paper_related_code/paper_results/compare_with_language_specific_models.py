import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

target_lgs = "af am	ar	as	ast	az	be	bg	bn	bs	ca	ceb	cs	cy	da	de	el	en	es	et	fa	ff	fi	fr	ga	gl	gu	ha	he	hi	hr	hu	hy	id	ig	is	it	ja	jv	ka	kam	kea	kk	km	kn	ko	ku	ky	lb	lg	ln	lo	lt	luo	lv	mi	mk	ml	mn	mr	ms	mt	my	ne	nl	no	ns	ny	oc	om	or	pa	pl	ps	pt	ro	ru	sd	sk	sl	sn	so	sr	sv	sw	ta	te	tg	th	tl	tr	uk	umb	ur	uz	vi	wo	xh	yo	zhtrad	zu".split()
zh2x_chinese_llama2 = "1.59	0.15	0.45	0.1	4.77	0.58	0.62	1.43	0.13	2.05	13.5	1.57	1.28	0.82	4.16	11.26	0.39	30.69	15.14	0.78	0.69	0.82	1.38	19.28	1.36	3.33	0.14	0.83	0.33	0.26	1.94	1.87	0.14	5.18	0.88	0.61	10.52	6.81	1.9	0.22	0.81	1.56	0.32	0.56	0.15	1.63	0.16	0.28	1.38	0.88	0.92	0.67	0.58	0.61	0.46	0.95	0.7	0.06	0.16	0.19	3.1	1.22	0.07	0.45	3.97	3.4	1.1	0.74	3.01	0.32	0.17	0.12	3.2	0.42	12.98	2.95	3.42	0.17	1.24	1.11	1.17	0.9	0.35	2.22	0.58	0.16	0.25	0.25	0.42	1.36	1.16	1.56	0.72	0.13	0.45	7.19	0.89	0.74	0.58	5.11	0.73".split()
zh2x_xllama2 = "19.11	10.5	15.82	10.88	15.53	12.43	14.21	21.01	17.8	16.04	24.57	16.73	18.1	20.82	21.15	19.72	15.48	28.71	18.7	15.02	15.11	1.05	16.11	25.59	17.4	20.13	13.55	10.27	16.54	19.09	16.55	16.61	18.1	21.59	11.59	15.54	19.47	18.75	13.14	17.94	1.87	2.35	15.49	8.73	12.65	16.2	1.31	15.23	12.66	1.63	4.43	6.28	14.52	1.69	16.26	12.09	20.44	7.47	10.8	13.86	19.81	25.22	12.11	16.89	18.18	16.88	4.02	10.83	17.3	0.78	10.18	14.87	15.8	10.53	23.71	17.69	17.14	16.25	16.24	17.56	9.34	7.56	4.79	20.58	17.26	15.14	13.5	17.49	12.65	18.87	14.16	17.3	0.89	13.86	14.17	24.85	0.68	9.61	2.48	11.09	10.69".split()
zh2x_llama2 = "0.72	0.07	0.34	0.09	0.68	0.47	0.04	0.49	0.18	0.83	1.52	0.5	1.04	0.46	2.07	2.43	0.23	13.79	7.24	0.65	0.6	0.6	1.13	4.58	0.18	0.85	0.15	0.43	0.32	0.19	0.97	0.88	0.11	1.5	0.23	0.6	2.16	0.8	0.47	0.19	0.48	0.47	0.15	0.47	0.15	2.22	0.1	0.1	0.38	0.58	0.61	0.51	0.49	0.55	0.38	0.34	0.26	0.05	0.11	0.13	0.7	0.13	0.03	0.39	2.71	1.22	0.39	0.24	0.45	0.04	0.12	0.13	1.07	0.3	4.04	0.77	0.69	0.14	0.64	0.71	0.29	0.16	0.26	1.63	0.46	0.16	0.25	0.14	0.42	0.55	0.67	0.39	0.36	0.11	0.29	2.29	0.4	0.29	0.22	7.67	0.26".split()

ja2x_jpnllma2 = "0.8	0.1	0.13	0.1	0.9	0.3	0.13	0.15	0.07	0.52	1.43	0.88	0.69	0.74	0.95	1.8	0.19	7.88	1.9	0.52	0.16	0.69	0.46	3.57	0.71	1.03	0.12	0.65	0.18	0.14	0.54	0.6	0.09	0.87	0.68	0.56	1.87	0.58	0.16	0.74	0.7	0.21	0.42	0.15	1.18	0.09	0.14	0.65	0.77	0.66	0.66	0.35	0.81	0.27	0.92	0.17	0.12	0.1	0.1	0.77	0.82	0.16	0.24	1.42	0.9	0.89	0.73	0.87	0.23	0.11	0.14	0.57	0.23	1.35	0.78	0.34	0.13	0.48	0.55	0.76	0.64	0.11	1.03	0.52	0.13	0.25	0.17	0.2	0.98	0.56	0.15	0.46	0.09	0.4	0.72	0.68	0.56	0.63	2.58	1.38	0.44".split()
ja2x_xllama2 = "16.91	10.79	14.88	10.29	14.71	11.94	13.98	18.26	18	14.41	22.68	15.49	15.82	19.54	19.77	18.08	15.23	26.3	18.21	14.12	14.12	1.02	14.48	24.29	16.26	18.86	12	8.87	15.68	17.97	14.98	14.42	16.67	19.29	9.19	13.97	18.21	10.48	17.43	1.53	1.62	14.82	9.34	12.25	19.75	1.15	15.19	10.79	1.26	3.53	5	12.86	1.54	14.66	11.5	19.16	10.19	10.92	13.17	18.03	24.06	12.19	16.34	17.2	16.19	3.55	9.38	14.76	0.72	8.5	12.44	14.77	9.8	21.96	17.46	16.74	15.38	13.57	15.53	7.69	6.45	3.54	18.8	15.62	14.75	13.2	15.73	11.04	18.16	13.27	16.67	0.55	13.81	12.93	21.73	0.56	7.9	2.22	13.93	6.11	8.34".split()
ja2x_llama2 = "4.32	0.05	2.62	0.2	5.17	0.67	0.56	8.65	1.69	6.3	16.65	1.23	7.75	1.04	9.41	12.33	1.52	22.89	14.14	1.16	2.47	1.13	7.45	18.87	1.02	5.38	0.88	0.53	1.11	4.22	7.32	8.5	1.27	10.96	0.41	0.8	12.78	1.88	1.58	1.13	0.91	0.58	0.2	0.53	15.23	0.1	0.44	0.69	1.29	0.78	0.2	0.85	1.43	0.62	0.92	2.9	0.47	0.23	1.05	7.02	1.1	0.09	1.3	11.39	6.78	1.2	0.76	3.56	0.18	0.45	0.4	8.7	0.32	15.42	7.33	11.2	0.29	2.81	6.84	0.95	0.55	0.72	7.75	0.47	0.57	0.64	0.37	0.88	2.87	2.2	10.9	0.58	0.73	0.41	15.13	0.71	0.55	0.61	7.45	4.31	0.38".split()

zh2x = [zh2x_chinese_llama2, zh2x_xllama2, zh2x_llama2]
jpn2x = [ja2x_xllama2, ja2x_llama2, ja2x_jpnllma2]

zh2x = [list(map(float, single_l)) for single_l in zh2x]
jpn2x = [list(map(float, single_l)) for single_l in jpn2x]


x2zh_ChineseLLaMA2 = "9.62	0.27	4.92	0.22	12.29	2.06	1.6	11.37	0.31	12.54	19.2	4.57	14.1	2.1	16.13	19.69	2.62	29.94	14.22	3.55	2.58	1.12	14.32	21.52	1.94	15.31	0.2	1.39	0.61	1.27	12.95	13.2	0.41	16.88	1.19	2.73	18.97	13.13	1.78	0.38	1.7	5.27	1.66	0.99	0.36	11.47	0.26	1.3	4.67	1.1	1.61	1.96	2.32	1.22	2.37	1.18	8.28	0.09	1.11	0.43	10.48	2.94	0.19	0.63	14.4	13	1.56	1.81	7.88	0.58	0.33	0.37	13.21	0.73	17.19	17.41	19.23	0.45	11.17	12.42	1.12	1.34	10.28	18.8	1.54	0.3	0.65	0.91	1.39	6.88	5.9	10.44	0.77	0.62	1.49	13.74	1.38	1.46	1.27 21.72	1.02".split()
x2zh_xllama2 = "18.28	8.44	15.68	9.32	16.15	13.24	13.7	17.67	13.73	17.01	16.86	15.29	16.99	15.68	18.52	18.63	15.51	22.4	16.48	16.42	15.04	2.05	16.35	18.08	13.49	17.72	12.22	11.4	15.55	15.21	16.45	16.84	15.48	18	8.29	14.21	16.89	13.93	13.39	13.88	2.44	7.71	15.55	12.93	11.25	15.88	9.73	12.95	15.83	4.53	6.87	9.68	15.01	2.47	15.92	8.43	18.63	12.64	13.29	13.88	17.76	17.37	8.12	14.42	15.65	17.79	6	9.85	16.62	2.5	9.27	12.27	15.1	12.48	18.28	17.6	17.84	11.43	17.01	17.06	10.37	10.14	17.63	19.1	14.57	11.01	11.73	15.34	12.9	15.09	14.99	17.29	1.97	13.16	14.3	15.68	3.6	11.46	5.14	19.44	11.58".split()
x2zh_llama2 = "7.34	0.13	4.78	0.43	6.92	2.35	3.01	9.67	1.25	9.47	8.39	3.14	8.59	2.37	9.58	11.52	4.12	16.6	8.55	3.87	4.57	1	9.84	11.13	2.13	8.09	0.3	0.95	2.64	4.44	9.2	7.56	0.68	10.09	0.77	3.32	9.67	7.45	2.36	1.41	1.03	3.85	1.72	0.88	0.26	9.08	0.45	1.4	3.54	1.07	1.09	0.84	3.17	1.04	2.55	1.49	8.62	0.28	0.87	1.4	9.49	3.18	0.17	2.06	8.47	10.2	1.11	1.24	6.49	0.42	0.27	0.26	9.21	0.7	9.69	10.49	11.84	0.55	7.21	9.63	0.95	0.96	9.83	11.04	1.93	0.37	0.36	0.98	2.44	4.02	5.92	10.64	0.79	1.62	1.43	7.62	1.34	1.2	0.88 18.47	0.94".split()


x2ja_xllama2 = "20.08	9.96	16.98	12.22	19.04	15.3	15.17	19.86	15.49	18.88	19.38	18.35	18.14	18.03	21.04	22.12	18.31	26.03	18.12	17.48	17.7	3.25	18.64	20.93	14.9	20.41	15.33	13.83	16.99	16.24	18.09	17.84	15.65	20.47	10.02	17.47	20.01	16.3	15.36	3.17	10.35	18.76	10.3	14.88	3.19	11.71	16.25	18.12	6.18	9.01	9.61	17.01	2.85	18.25	11.03	20.25	15.6	16.4	16.52	20.21	19.61	9.27	15.77	18.67	19.83	7.77	12.79	18.92	3.91	12.54	14.66	16.69	14.31	20.85	19.3	19.97	13.58	17.37	18.23	12.22	11.88	20.28	21.19	17.07	13.38	14.02	17.18	13.02	18.57	16.4	18.57	2.79	15.96	16.95	15.11	4.5	14.07	6.19	18.75	17.99	13.09".split()
x2ja_llama2 = "6.16	0.05	2.35	0.55	5.9	2.23	1.57	5.13	1.88	5.45	4.67	3.18	4.85	2.13	6.6	8.96	2.69	20.78	3	2.99	3.42	1.56	7.53	7.27	2.09	4.88	0.48	1.28	2.05	2.19	5.05	3.35	0.95	8.05	0.91	2.95	6.75	2.62	1.04	1.39	3.35	1.93	0.5	0.35	1.83	0.57	1.76	3.28	1.32	1.45	0.93	2.17	1.37	1.75	1.43	4.83	0.68	1.36	1.07	8.02	3.29	0.11	1.09	7.11	7.16	1.25	1.65	5.18	0.42	0.25	0.35	5.02	0.85	6.7	5.65	8.13	0.59	2.22	4.56	1.01	1.33	6.38	7.7	1.87	0.36	0.43	1.33	1.2	2.67	4.23	4.85	0.84	1.3	1.68	1.66	1.23	0.91	0.75	3.28	3.99	0.85".split()
x2ja_jpnllma2 = "4.18	0.08	0.27	0.2	4.03	1.26	0.92	3.69	0.33	3.68	5.59	2.17	4.94	0.93	6.02	7.83	0.33	12.17	5.19	1.9	0.43	1.08	3.5	6.92	1.08	4.54	0.11	0.99	0.13	0.41	3.84	4.05	0.28	5.24	0.94	1.33	6.31	1.33	0.28	1.08	2.35	1.14	0.38	0.19	2.56	0.12	0.92	2.06	1.03	1.18	0.25	1.4	1.07	1.29	1.02	2.75	0.12	0.86	0.21	4.1	1.8	0.26	0.27	5.57	5.58	1.07	1.1	4.15	0.52	0.07	0.03	4.74	0.23	6.02	5.29	4.74	0.11	3.35	3.33	0.98	1	3.52	6.19	1.24	0.19	0.56	0.65	0.34	2.98	2.58	4.25	0.85	0.21	1.01	3.22	1.01	1.07	0.99	2.49	2.32	0.79".split()


x2zh = [x2zh_ChineseLLaMA2, x2zh_xllama2, x2zh_llama2]
x2jpn = [x2ja_xllama2, x2ja_llama2, x2ja_jpnllma2]

x2zh = [list(map(float, single_l)) for single_l in x2zh]
x2jpn = [list(map(float, single_l)) for single_l in x2jpn]



label_zh = ["ChineseLLaMA2-7B-Alpaca", "LLaMAX2-7B-Alpaca", "LLaMA2-7B-Alpaca"]
label_ja = ["LLaMAX2-7B-Alpaca", "LLaMA2-7B-Alpaca", "Swallow"]


df = pd.DataFrame(columns=["model", "Language", "BLEU", "direction"])

for i,  model in enumerate(label_zh):
    for lg, s in zip(target_lgs, zh2x[i]):
        df.loc[df.shape[0]] = [model, lg, s, "zh $\\rightarrow$ X"]

for i,  model in enumerate(label_zh):
    for lg, s in zip(target_lgs, x2zh[i]):
        df.loc[df.shape[0]] = [model, lg, s, "X $\\rightarrow$ zh"]


for i,  model in enumerate(label_ja):
    for lg, s in zip(target_lgs, jpn2x[i]):
        df.loc[df.shape[0]] = [model, lg, s, "ja $\\rightarrow$ X"]

for i,  model in enumerate(label_ja):
    for lg, s in zip(target_lgs, x2jpn[i]):
        df.loc[df.shape[0]] = [model, lg, s, "X $\\rightarrow$ ja"]


# df = df.sort_values("model")


sns.set(style="white")
flatui = ["#2ecc71", "#EA4335", "#4285F4", '#FBBB01']
sns.set_palette(flatui)

plt.figure(figsize=(10, 6))
# sns.lineplot(x="Language", y="BLEU", hue="model", data=df, dodge=True)
sns.boxplot(x="direction", y="BLEU", hue="model", data=df)
# plt.ylim((10, 25))
plt.ylim((-2,32))
plt.legend(loc="best", ncol=2, fontsize=16)
# plt.xticks([x for i, x in enumerate(target_lgs) if i % 2 == 0], ha='center', fontsize=18, rotation=45)
# plt.xticks(target_lgs, ha='center', fontsize=18, rotation=45)
plt.yticks(fontsize = 25)
plt.xticks(fontsize = 25)
plt.xlabel(None, fontsize = 25)
plt.ylabel("spBLEU", fontsize = 25)
plt.tight_layout()
# plt.show()
plt.savefig("./comparison_lg_specific_llm.pdf", dpi=600)





