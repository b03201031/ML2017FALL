import numpy as np
import pandas as pd
import derivative as fc
import sys


f_path_in = sys.argv[1]
f_path_out = sys.argv[2]

NUM_DURATION = 9
NUM_ELEMENT_SELECTED = 7
data_selected = np.array([0, 2, 5, 7, 8, 9, 12])
#full_beta = pd.read_csv("beta.csv", header = None)
#full_beta = full_beta.apply(pd.to_numeric)
beta = np.array([21.540614414209013,-0.21150054746624417,-0.1587348111833179,0.47968045963518774,-0.6632285837652029,-0.06196288441963403,-0.19592053282095995,0.289877699668566,0.09542361968660562,1.113640501807589,-0.12268569437591798,1.1421259868671625,-0.4297058740519056,-0.09300798116468581,-0.12635309996192481,-0.426899765777186,-0.3470105527994124,0.24398296329278352,1.4833516707613361,-0.3448233988215449,-0.6210978086288084,0.1993324526070583,0.06147109622814125,0.34492656530468235,0.13064360493367633,-0.2958910729221602,-0.32815078748438153,1.2028940858014507,1.0939107304119449,-0.6906641623740928,-0.7907130036655017,0.12825347405873797,-0.10947399939503678,0.22483027664486593,-1.346896852926156,-0.2688181709361373,2.782044103500695,0.14809880787310656,-0.11019696599588005,-0.7466557695390249,1.0854104657042862,0.45133398013165393,-1.0344400650240606,0.505876454130266,0.8055756445595673,0.476127111995696,0.20987186482008685,-0.009606054783795467,2.331046938599736,-2.979543523853721,0.34811283112668934,5.535327250083149,-6.720073412086077,1.3925293621492414,13.671885186399209,-0.48460161416913666,0.9902307126209906,-0.14032642434780027,-0.2256642826321732,0.16486700960770867,0.6099582183125676,-0.8844593721753775,-0.008999307319683129,1.2939614061186662,-0.17984722834781722,-1.2135602893911464,1.4020225141101021,-0.21227262120084908,0.5143864183062747,0.6941326686621317,0.10796703670224884,-2.176151753367971,0.013908550542481296,-0.019411772462923763,-1.01214479904698,0.26113113134868016,0.12721919812579846,0.27346550890991994,0.31761157483234176,0.13943313717497574,-0.30551009386371786,-0.9845015384270696,0.6485698463660976,-0.33385289558443315,-0.22398819167041217,-0.1871423899177669,-0.08796954483926499,-0.033810191076283407,-0.3299557189250443,0.10630804086834847,0.306497342094914,-0.719865248309374,0.5555863456371475,0.295654810385309,-0.2378229594770259,0.15735239711287102,-0.554153381412038,0.6693954692139487,0.07381447785713541,-0.8088175012230652,0.11656245672741487,0.3318360798910543,0.0015220622779837038,0.046926259846836314,-0.8321513670985536,0.10430059174429487,0.6100875782065329,-1.1707960441846932,0.6043446608031545,-0.999664022976823,0.08139685073187264,0.9805715334954591,-0.8317965701559594,-0.33647212372850344,2.105513818107736,-2.433912702353776,-0.39494081995608726,1.8362993460046737,-0.03864021888282837,-0.23856537008037346,0.020446930028330697,0.08519273019894401,-0.19909476193890377,-0.40926501323857256,0.5702069335992281,0.15122646801077463,-0.967607909586568])

full_data = pd.read_csv(f_path_in, header = None)
full_data.drop(full_data.columns[0:2], axis=1, inplace=True)
full_data.replace({'NR':0}, inplace=True)
full_data = full_data.apply(pd.to_numeric)

tmp_ls_data = [full_data[i*18: (i+1)*18].reset_index(drop=True) for i in range(int(len(full_data)/18))]

x = np.array([[]])



#x.shape = int(len(full_data)/18), NUM_ELEMENT_SELECTED*NUM_DURATION

for df in tmp_ls_data:
	for i in data_selected:
		x = np.append(x, df.values[i, 9-NUM_DURATION:])
	

x.shape = int(len(full_data)/18), NUM_DURATION*NUM_ELEMENT_SELECTED


x = np.concatenate((x,x*x), axis=1)
#x = np.concatenate((x,x*x*x), axis=1)

M = np.array([23.148413926499032,23.157117988394585,23.166402321083172,23.17452611218569,23.180328820116056,23.183617021276596,23.185628626692456,23.187001934235976,23.185338491295937,0.3876634429400387,0.38749129593810444,0.387431334622824,0.38746615087040626,0.3876189555125725,0.38776015473887815,0.3877272727272727,0.3875222437137331,0.3870135396518375,9.9847582205029,9.985725338491298,9.99191489361702,10.002785299806577,10.016460348162475,10.029071566731142,10.03340425531915,10.034081237911025,10.027388781431334,32.373230174081236,32.42139264990328,32.46628626692456,32.50825918762089,32.536924564796905,32.5558413926499,32.5704835589942,32.58203094777562,32.59148936170213,42.80174081237911,42.811218568665375,42.821083172147,42.84139264990329,42.870793036750484,42.915667311411994,42.95473887814313,42.9926499032882,43.01237911025145,21.51779497098646,21.51044487427466,21.499419729206963,21.50193423597679,21.512185686653773,21.52707930367505,21.536170212765956,21.541586073500966,21.542166344294003,2.8146228239845263,2.8154352030947773,2.818878143133462,2.8221856866537713,2.8243713733075437,2.8262669245647967,2.827214700193424,2.827678916827853,2.8289941972920696,573.3837234042553,573.8051934235976,574.2477466150871,574.6481334622824,574.935174081238,575.1055802707931,575.2234448742746,575.3079342359767,575.2536905222437,0.2647966731141199,0.2646392069632495,0.26457887814313347,0.2645983752417795,0.2647187814313346,0.26483423597678923,0.2648010251450677,0.26461769825918763,0.26412088974854936,138.65795357833653,138.6468820116054,138.80141586073503,139.08971373307546,139.4416421663443,139.73631528046423,139.7982998065764,139.78679690522245,139.6139845261122,1417.1059613152804,1421.0496750483558,1424.564760154739,1427.7251083172148,1429.6069574468083,1430.707839458414,1431.3949013539652,1432.0219245647968,1432.577667311412,2538.1235976789167,2538.325338491296,2538.3781431334623,2539.936170212766,2541.976015473888,2546.332688588008,2549.404255319149,2552.071566731141,2553.0785299806575,749.6574468085106,748.611411992263,747.2181818181818,746.5177949709864,746.5458413926499,746.9723404255319,747.2143133462282,747.2297872340425,747.0013539651837,11.27793423597679,11.280932301740812,11.316580270793038,11.347375241779497,11.359555125725338,11.369864603481625,11.373444874274663,11.37336750483559,11.38731334622824])
SD = np.array([6.126553361533744,6.128056779695293,6.129074164406777,6.131025439345905,6.132497849370636,6.133961369422393,6.135965291571807,6.137660428674301,6.139525190621467,0.33839877086360043,0.33836327007442646,0.3383427834246075,0.33833172652213833,0.33833463724347823,0.33834346214116484,0.33833206636164315,0.33829603734933106,0.3381440667447863,6.241999427782504,6.239565006939003,6.242039139532129,6.247719582299453,6.253971878736984,6.257318817980141,6.255325639478688,6.2533199675232085,6.250236695688956,19.211453079120183,19.23286181745066,19.248506856087385,19.259755785255408,19.260205017380397,19.256817771247587,19.249896157830285,19.24664084672929,19.245064003465078,26.573192903917896,26.561718753824174,26.54680732399865,26.543384223372545,26.53433850454927,26.543891723035056,26.538550511090204,26.52741263343107,26.514406899960267,16.930503430054948,16.908937675177953,16.881739635640994,16.857776219973847,16.845524876820445,16.8391566617458,16.834716686448935,16.828245799012826,16.82071410969081,1.8318931722898437,1.8314630542043524,1.835893865384776,1.8391963440117127,1.8391523785065211,1.8390431954123987,1.8385597388403867,1.8378246510923733,1.839593753499949,262.5485985913557,262.65011836743355,262.72848899302886,262.8561432941619,262.92646468879985,262.97258303964,263.0253342008699,263.05542713226475,263.0756290604777,1.496974989653645,1.4969732230247492,1.4969715906242365,1.4969709652927787,1.4969674693161508,1.4969658550998048,1.496967022580449,1.4969704949880354,1.49694996180622,203.76503105014763,203.71482874979935,203.84884428187303,204.0795923301536,204.33078632327624,204.44981967875168,204.43117749043896,204.40932530942436,204.34200297546184,1919.355897962862,1921.895796082701,1923.6850801835717,1924.7473028823301,1924.9169027782598,1924.8412750101377,1924.6906343050925,1924.62823457035,1924.554097676704,3283.9809239257133,3280.3998176727628,3274.666469756694,3270.946882969255,3266.397431020078,3266.5345875941234,3265.261432571365,3264.725221523914,3263.97596175881,1183.2894133116406,1178.7997118275805,1172.9047575108857,1168.7694819371736,1167.0581996293624,1165.8482477402877,1164.756493635604,1163.7308919923462,1163.1209397569505,19.420216899836316,19.417062484658594,19.504130929501162,19.562596678286177,19.56226296067282,19.562030783542163,19.56065815003265,19.557781630243163,19.58356007717711])

for i in range(len(x[0,:])):
	x[:, i] = (x[:, i] - M[i])/SD[i]
	


x = np.concatenate((np.reshape(np.repeat(1, len(x)), (len(x), 1)),x), axis=1)
#print(x)


y = np.array([[]])


col_beta = np.reshape(beta, (len(beta), 1))
for x_0 in x:
	#print(np.dot(x_0, col_beta))

	y = np.append(y, np.dot(x_0, col_beta))




df_y = pd.DataFrame(y)
df_y.columns = ["value"]



ls_id = []
for i in range(len(df_y.index)):
	ls_id.append("id_"+str(i)) 

df_y.index = ls_id
df_y.rename_axis = "id"


df_y.to_csv(f_path_out, index_label = "id")