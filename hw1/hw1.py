import numpy as np
import pandas as pd
import derivative as fc
import sys


f_path_in = sys.argv[1]
f_path_out = sys.argv[2]

NUM_DURATION = 9
NUM_ELEMENT_SELECTED = 18
data_selected = np.arange(18)
#full_beta = pd.read_csv("beta.csv", header = None)
#full_beta = full_beta.apply(pd.to_numeric)
beta = np.array([21.37163030008338,0.2348133538863105,-0.49353350532881773,0.3311687926055854,-0.868086208927129,0.15107953087387405,0.6399899784930835,-0.11580260700954023,-0.49982420433497776,1.3660707946857509,-0.14933901242952002,0.22762921410806253,0.13481395145124553,0.14381774114244855,-0.11260008482996167,0.2535785369132543,0.23056557635675,-0.022496091117328882,-0.7404210832347718,0.0025874297943938042,1.0351419088873948,-0.43540241102493693,-0.052172533498337136,-0.053267883582453525,-0.17592803745705288,-0.4994050415498181,0.11437549859972428,1.4404484167000042,-0.4981297134915898,0.27367170998083107,0.37731532105655324,0.09810685284650962,-0.218520308030564,-0.09732552677416718,0.06966761410760468,0.215047237883584,-0.2769021608565168,-0.19744920653437076,0.07664476466096866,0.17832984494835902,0.18950969701798215,-0.624650365854503,0.23426127207826672,0.38595772424430247,-0.3329042500454657,-0.6682217055739952,0.06133567162700583,-0.378112815600456,-0.014357320083216907,-0.2741623137089742,0.07478667900719936,-0.2244556789617796,-0.21702523706285665,-0.30154661450371073,0.4900110842974289,0.06526807130328918,-0.2945251885408125,0.15906075061447142,0.44888507435616853,0.38000018942662905,0.05343862606845631,-0.0557913051476266,0.2937554896711793,0.46225794324532543,0.4904165946052235,-0.3732304092456733,-0.23318783238563248,-0.15796829901786172,-0.04966486570392185,0.18305772542211698,-1.3557246018279938,-0.4400594898498067,3.01149125599948,0.15212939150023638,-0.22019627438079212,-0.4018172028949255,0.9168876309417813,-0.05185236114469868,-0.5772726311302597,0.46911481835317115,0.3079659662939209,0.7990075027798275,0.03292623134223842,-0.05810346955540496,1.7437387087899032,-2.311566709107061,0.45178586824190775,4.51995384408515,-5.719922186110285,1.428447483020918,12.762714897483846,-0.05356509136219791,-0.24857966704117743,-0.10454054939056223,0.1761154691353143,-0.2249536209383156,0.11795366194568575,0.09733501267787344,-0.2657244132647082,-0.39586719522120706,0.6065934909164571,0.36099929319266455,-0.22334566446065413,-1.3043370950949973,-0.5089465452156653,0.8083205109729051,-0.5834911184066425,0.25663916400476977,1.2377173393836793,-0.7328757459242382,1.0199012829974816,-0.26977964171965463,-0.23471240098320503,0.13226910689398133,0.4597493912281305,-0.7756522499873032,0.020543419732898652,1.219338953092959,0.4860041895696684,0.10398639335950956,-0.04978735404374675,-0.4008546986189218,-0.1167232517074533,0.15840320368050104,0.32892361235719175,-0.07939145044220498,-0.5448181113055865,-0.5023493157860858,0.8536585405050903,0.3028156620400382,-0.40467164895832425,-0.7713404650498934,0.17122305872193613,0.3258360111273756,7.815870890268549e-05,0.18828992690825466,0.3207918236900995,-0.476060057459913,0.11032708149260362,-0.2598206638969998,0.5402998151737478,-0.11337638347607644,0.9087982242594181,0.6959290076286732,1.120776894289985,-0.023574499482352906,-0.8479817353339859,0.14993823660093705,-0.17682312386142893,0.2321604359039888,-0.35403289432503243,-0.01848320866236977,-0.18333892041035565,-0.35216444890092186,-0.15004180095379394,0.354678064414454,0.20107616779515955,0.12739872738950492,0.25465862884577833,0.16245516689680228,-0.0868246349942975,-0.27959948753672415,-0.09739030083182496,0.3459007950430558,-0.6973645185017763,0.5670995287181562,-0.6267076725600642,0.36345273991554894,0.765263439034852,-0.7147037414191116,-2.093989184937525,1.1620334449882976,-0.15917732744281268,-0.19011076909332247,-0.12241210994838758,-0.12924292303331336,0.15858079007643883,-0.2790162432045013,-0.5245694535710491,0.1530237223550901,1.4663572563979865,-0.0966756810628756,-0.883807498649159,0.3298126296873018,0.10195361410820208,0.1736334038520291,0.15997165625235235,0.2594412366510638,-0.21939869472610254,-0.9472913793064306,0.46805567704609846,0.0984269670459289,-0.45377941184759407,0.24148967605111446,0.42342974352830487,-0.23691864774135887,0.07490728814472358,-0.05357876268066629,-0.307067128891896,0.315411254308909,0.06286652463271052,0.19484713656192404,-0.5432036877556424,0.43912863621422954,-0.4624380639787854,-0.3863847323692112,0.12889032739837814,0.3424789180554133,0.5213059348535528,-0.057732817236594965,0.03634430110852185,-0.6517106044709016,-0.15402857645550094,-0.1069236466235701,-0.3994478695679802,-0.7956474380864642,0.3008441267031826,-0.20293606946777026,-0.2623924887795239,-0.28872460591293336,0.4606083624468324,-0.07572528334865153,0.3886288265219651,0.379734114290456,0.793610498794366,0.15666570106751232,-0.47184961583651447,0.4257907073409132,0.013227591214832853,-0.06434565983591424,0.013694475550816128,-0.42293622382902124,0.6024996570387466,0.15669159578972985,-1.0023711431443594,0.05886416551685976,0.31191829809988664,0.014457795102416633,0.010132804332829511,-0.5777247855641345,0.022916319207320034,0.5018677664130614,-0.7842528885258729,0.32483031218759184,-0.7336806310014741,0.18191964642536868,1.1731974589906762,-1.176803848468587,-0.14924881060407505,2.6836827221328634,-3.2627130025154965,-0.4093825874473569,2.463271237753903,0.14020385999604243,0.2292098210443526,0.04137173955192985,-0.16736812415927993,0.10111345749254226,0.04044369257775351,0.012619424234479761,0.20562013919604427,0.29258891200536974,-0.3316405068717108,-0.23990053572046977,0.31597090467314054,0.4351709005739666,0.2304373929367589,0.4260874493914371,-0.7670579366271825,-0.7225755138908443,-0.3666695650554614,0.09135949628959945,-0.285981534761911,0.09500227481342532,0.10473854692882772,-0.18440125557977574,-0.2992606459710108,0.46373588618901346,0.12772502451912754,-0.8222030911411439,-0.46638507462941237,-0.47169123968947685,0.20038106390830904,-0.05900371900248944,0.04599177514030401,0.16135609610477028,-0.36634819734613117,-0.3161995010894952,0.9869528865501856,0.5137658855528243,-0.5390417577641644,-0.39190227206700096,0.5856418856608884,0.8118071041388333,-0.015545696191581703,-0.5431087476960399,0.1545394660982662,-0.15558272325658504,-0.5348595939866753,0.42089164940017204,0.006543078896329085,0.08429244339928652,-0.5213108262428193,0.13570443656541714,-0.9021708408551551,-0.8814987235617974,-1.0901507341437549,-0.12612410413483102,0.8204470136753705,0.1649800959521622,0.3230168564938309,-0.384409999910747,0.31559947182216563,0.11323724918066871,0.19937453912176806,0.17384964212325876,0.0695698818243909,-0.1919826606828994,-0.4467637910193202,-0.47940244054258635,-0.3699001665538708,0.267132747607575,-0.09371588199424734,-0.1431134767232563,0.23720989154481947
])

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

M = np.array([22.531099290780144,22.538368794326242,22.54599290780142,22.552375886524825,22.55663120567376,22.558581560283688,22.559184397163122,22.559202127659574,22.556436170212766,1.7024468085106381,1.702340425531915,1.7022340425531912,1.702127659574468,1.7020212765957445,1.7019326241134751,1.7018794326241136,1.7018439716312057,1.701790780141844,0.38924468085106384,0.3890904255319149,0.38904787234042554,0.38910283687943265,0.3892783687943262,0.38943439716312056,0.38940248226950364,0.38919326241134755,0.38870035460992913,0.1401808510638298,0.1401950354609929,0.14025531914893616,0.1405354609929078,0.1408741134751773,0.1410904255319149,0.14117375886524824,0.1411613475177305,0.14100709219858157,2.1482978723404256,2.1499468085106384,2.153014184397163,2.1551063829787234,2.1571276595744684,2.157712765957447,2.157021276595745,2.155921985815603,2.150904255319149,10.116932624113476,10.11781914893617,10.123829787234042,10.134929078014185,10.14959219858156,10.163404255319149,10.16845744680851,10.169645390070922,10.164095744680852,12.251170212765958,12.253173758865247,12.262446808510639,12.275780141843974,12.29281914893617,12.307163120567376,12.311737588652482,12.31171985815603,12.301241134751773,31.863563829787235,31.911436170212767,31.956312056737584,31.996560283687938,32.02159574468085,32.038226950354606,32.051294326241134,32.06170212765957,32.070195035461,42.56932624113475,42.57340425531915,42.581382978723404,42.60070921985815,42.62943262411348,42.672340425531914,42.71152482269503,42.75336879432624,42.7781914893617,21.336524822695036,21.326063829787234,21.31418439716312,21.3177304964539,21.329787234042552,21.347340425531915,21.35762411347518,21.365957446808512,21.368794326241133,0.20411347517730496,0.20358156028368796,0.20312056737588652,0.20304964539007092,0.20287234042553193,0.2026595744680851,0.20258865248226948,0.2025531914893617,0.2025531914893617,73.20939716312057,73.16719858156029,73.12641843971632,73.09113475177305,73.06400709219858,73.04751773049645,73.03599290780141,73.02783687943263,73.02872340425532,2.759804964539007,2.760691489361702,2.764060283687943,2.7671985815602835,2.7693617021276595,2.771436170212766,2.772517730496454,2.7731028368794326,2.774468085106383,1.839450354609929,1.8393794326241135,1.8393262411347517,1.8395035460992906,1.8397695035460992,1.8398758865248226,1.839875886524823,1.8397517730496455,1.8395567375886526,156.1201418439716,156.3508156028369,156.59,156.79195035460992,156.7595035460993,156.76216312056738,156.73751773049645,156.7814893617021,156.84159574468086,158.49271276595744,158.63296099290778,158.82835106382979,158.96214539007093,158.9288120567376,158.92845744680852,158.9866134751773,159.03413120567376,159.0720744680851,2.2951773049645388,2.298262411347518,2.3004787234042556,2.3027482269503547,2.3035460992907804,2.3050886524822696,2.3058510638297873,2.306099290780142,2.3065425531914894,1.7132978723404255,1.714290780141844,1.715017730496454,1.7148936170212765,1.7150709219858156,1.7154787234042552,1.71572695035461,1.7158687943262412,1.7167021276595746,547.1629468085107,547.5195070921986,547.8888333333333,548.2133014184398,548.4359964539007,548.5539042553191,548.6185070921986,548.6549982269504,548.5643173758865,2.9141914893617025,2.913822695035462,2.9134574468085117,2.9130921985815608,2.9127269503546103,2.9124343971631212,2.9122588652482273,2.912141843971632,2.9119627659574476,0.2580646453900709,0.2579219326241135,0.25787246453900714,0.2579016312056738,0.25804037234042554,0.2581714716312057,0.2581393085106383,0.2579487056737589,0.2574621808510638,0.0305759219858156,0.030575709219858156,0.03059113475177305,0.030721028368794333,0.030897464539007095,0.030989166666666672,0.03100843971631206,0.031000159574468087,0.030937659574468087,9.897691489361701,9.902179078014184,9.923460992907803,9.931751773049646,9.944031914893618,9.945725177304965,9.941741134751773,9.934925531914894,9.892455673758866,140.87039893617023,140.86025,141.0066560283688,141.28863120567377,141.66229609929078,141.9893829787234,142.06772517730496,142.0701170212766,141.92498758865247,208.02627304964543,208.04834929078012,208.31755319148937,208.7083865248227,209.20624645390072,209.5827411347518,209.66430141843975,209.64133510638297,209.31436170212766,1367.1656081560282,1370.9780195035462,1374.4124148936169,1377.4228758865247,1379.0746773049646,1380.05545035461,1380.6717819148935,1381.2414113475177,1381.7460549645389,2503.937765957447,2503.9290780141846,2503.9434397163122,2505.4085106382977,2507.398936170213,2511.5343971631205,2514.6498226950353,2517.747340425532,2519.088120567376,734.8730496453901,733.8285460992907,732.5088652482269,731.9177304964539,732.1003546099291,732.7877659574468,733.1324468085106,733.2712765957447,733.1102836879433,4.31241134751773,4.308943262411348,4.3066382978723405,4.3066099290780135,4.306432624113476,4.30617730496454,4.306148936170214,4.3061418439716315,4.3061418439716315,5539.603368794326,5533.472517730496,5527.434574468085,5522.031205673758,5517.726418439716,5515.017021276596,5513.033510638298,5511.595921985816,5511.670921985816,10.936310283687941,10.939682624113475,10.973296099290781,11.002120567375885,11.014450354609929,11.025551418439717,11.02968439716312,11.030171985815603,11.043546099290781,3.416828014184398,3.416540780141844,3.4163226950354613,3.417007092198582,3.4180549645390075,3.4184769503546106,3.418487588652483,3.4180283687943263,3.417278368794327,33529.292280141846,33624.06586170213,33721.3405070922,33805.51692553192,33804.41745744681,33808.909478723406,33804.11071985816,33825.896180851065,33844.65770567376,34091.14646631206,34149.062955673755,34231.160473404256,34282.59012588653,34277.93480673759,34280.73551595745,34303.12806914894,34326.59899113475,34337.04189893617,6.408872340425532,6.424017730496454,6.435246453900709,6.4465088652482265,6.448712765957446,6.456132978723404,6.458237588652482,6.45899645390071,6.460044326241134,4.072443262411348,4.075914893617021,4.07861524822695,4.08,4.082078014184398,4.082767730496453,4.083803191489362,4.084051418439716,4.086418439716312
])

SD = np.array([6.285897832252528,6.28819840519848,6.2902334721933455,6.2931425608619955,6.295584572152552,6.297960165271659,6.300929013010037,6.303760591135261,6.306465307523926,0.12596171463442812,0.12593558129170526,0.1259234338061445,0.1259111952651725,0.12589886564223549,0.1259354602220532,0.12595738191647435,0.1259719817822826,0.1259797864010749,0.32642491298800347,0.32639021643350297,0.3263651598537961,0.32634431745941067,0.3263475508246578,0.32636225568745475,0.3263510614552832,0.3263086118498408,0.32615060198806956,0.10452392539909573,0.10450388151618532,0.10449679517670951,0.10474164679011605,0.10512824878019081,0.10527420619453512,0.1052540237908686,0.10523133345846516,0.10514114096855139,2.2983706709448497,2.297804995770779,2.299563200847614,2.299406064855478,2.3001809005363842,2.300000173245498,2.2997826738769818,2.2993316252754235,2.2947911796520906,6.206293033310482,6.204029792757305,6.2060234021055845,6.210623462205441,6.216757539246878,6.220498124884029,6.218536671094771,6.216786119972758,6.214189028474549,7.611511115900746,7.60973600894518,7.612486549080898,7.616666490921802,7.621866164231001,7.62341636397349,7.621379072490856,7.619900868165511,7.615367899636871,18.758435654955377,18.778665049943807,18.793789788811242,18.80539831800561,18.806703147261107,18.80445330851666,18.798572124700232,18.79597478250991,18.794910091618718,26.301905431608503,26.295138868759395,26.282489671558842,26.279042688107698,26.270333273797267,26.279378983732464,26.274997050719982,26.26588656721606,26.250989571193074,16.72201417700365,16.704117684784492,16.67976045187557,16.657493718354278,16.64753829130318,16.64568480759803,16.642846482378395,16.636319874807473,16.62783547942263,2.066579066178823,2.0657922961235378,2.0652797226964608,2.065279828639627,2.0652543275835855,2.0652134034624923,2.065213493577991,2.065215254783029,2.065215254783029,13.415943343902425,13.417658866472104,13.414973002119813,13.405865371989687,13.393180580835583,13.381972003012182,13.370761033309932,13.361547915447458,13.35950897041458,1.822028221898256,1.8216105853555287,1.8257236503451721,1.8288610055405021,1.828957658722174,1.8288501778101307,1.8283407317143974,1.8275865620831233,1.82917274854426,0.18234694159706719,0.1822747458679854,0.18221326435955693,0.18230138805482562,0.18249147476182717,0.1825751147859603,0.18260424649218562,0.18259732296781161,0.18250856962488363,95.68591114035553,95.80442746569612,95.92138659909097,96.03020998378318,96.0774453523159,96.09648116598242,96.11171237440557,96.15331910224226,96.15285513148258,94.71645298679468,94.78737596482108,94.89317874193638,94.94012038581204,94.97140361693975,94.98674080910173,95.00728816697983,95.0512709162339,95.04271157409264,1.0681916865437795,1.0686475654225482,1.0691324038981613,1.0695136598123536,1.068825586519929,1.0689711328753748,1.0683110315294602,1.0681303829421005,1.0676637001219056,1.06632718386296,1.066359233437624,1.066456484020686,1.0673049621826445,1.0679933271060813,1.0676612196964794,1.0677472656561566,1.0676355647397195,1.0674044428450689,267.77535835642453,267.91584517060545,268.04255376742583,268.21544182872213,268.3314398625105,268.4190509687183,268.5143471196339,268.5893298813214,268.642798903604,0.32223667384412696,0.3221140228866989,0.32205434868725663,0.32199424911628916,0.3219337239355591,0.3220932411274029,0.32219532368705983,0.3222633076457164,0.322301775955142,1.4339110188306026,1.4339084180646395,1.4339057775145643,1.4339037374057895,1.433899166127135,1.4338976404756407,1.4338986437638606,1.4339007695291999,1.4338770468337676,0.05927905008015206,0.05927335379613429,0.05927197619906159,0.05944128506235647,0.059744586380085465,0.05981667985918624,0.05981421804675205,0.059812146305906984,0.05978231563436928,42.980780436199844,42.98001113852753,42.99591638625568,42.99557683562033,42.99821086355845,42.99796621207621,42.9980709783007,42.99741563108063,42.94627428149349,200.90846481563435,200.8619038736726,200.98280670808128,201.18829148929615,201.4280972018768,201.55232873988288,201.5307553583431,201.50994207743062,201.44845983570534,328.121620009459,328.08919652076037,328.2083055650548,328.39209045692195,328.56784973919315,328.6145345516852,328.58703542157184,328.5769868660196,328.45508537360786,1856.809671482229,1859.2545264173034,1860.985403506272,1862.050141113412,1862.2770386953528,1862.2476590161161,1862.129443468531,1862.0892196970613,1862.0359610664714,3211.2373014514674,3207.9966206924687,3202.646949888632,3199.155096093914,3194.8615079029673,3194.9941417533714,3193.7900698905482,3193.3139507036817,3192.4651028934204,1156.8004386261334,1152.6152122536346,1147.093828790434,1143.1965795403992,1141.6199226232602,1140.707407968872,1139.7185631560858,1138.7250531281936,1138.126400311386,106.43522414731821,106.43499341646628,106.43491475704018,106.4349158835854,106.43492222468403,106.43493082761485,106.43493195404632,106.43493223965059,106.43493223965058,1868.2164832593685,1867.9832519969714,1867.0759018794827,1865.372759817405,1863.1564557282613,1861.115649660834,1859.0567350379126,1857.3833680490714,1857.008920663542,18.833447703397994,18.830319096948124,18.912929837547335,18.96861711031296,18.968659353697454,18.968253498812963,18.966716211209537,18.96376686316931,18.988138473094434,0.6085835713595016,0.6082622921630767,0.607978444918008,0.6083948593752674,0.6093150675872193,0.609685044985234,0.6098083350690765,0.6097777870256098,0.6093766031746298,35832.565427171525,35884.84792954727,35937.165405265856,35982.14036660884,35983.14876592854,35989.36493521694,35991.84473945706,36016.02052444126,36022.05420025651,35423.56394983927,35451.92684296352,35495.87478229425,35515.7281537775,35521.55974913294,35526.76799061076,35544.9919130902,35571.66143806785,35580.32122264399,6.201399087567064,6.203519847895404,6.2052048339648005,6.20878471907865,6.206776765045907,6.209385157606888,6.207811880832693,6.206793300159825,6.206552361363475,5.164239134565653,5.164435965700538,5.164949051348893,5.168548851857987,5.1716116951902436,5.171343318626244,5.171443021957363,5.171106743363258,5.170875580075844
])
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
