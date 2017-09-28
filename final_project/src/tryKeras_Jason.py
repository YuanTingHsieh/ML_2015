import gensim
import numpy
#from combine import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.layers.recurrent import LSTM
dic = gensim.models.Word2Vec.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin', binary=True)
#from map_300to5 import *
__author__ = Jason_Wu

########################################### TRAIN ############################################################

'''
################ Question #######################
ques_files = open('dataset/pack/question.train.trim', 'r'); 
ques = [l.split() for l in ques_files.read().splitlines()]
# qv = numpy.array((len(ques), max_len), dtype=float32)
qv = []
for x in ques:
    m = []
    for g in x:
        if g in dic:
            m.append(dic[g])
    t = len(m)
    for _ in range(20 - t): # max_len = 20
        m.append(numpy.zeros(300, dtype=numpy.float32))
    qv.append(numpy.vstack(m))
#qv = numpy.stack(qv)
qv = numpy.dstack(qv).transpose(2,0,1)
max_len = max([len(l) for l in qv])
#for x in qv:
    #for c in x:
     #   print(c.shape)
ques_files.close()

################ Image ###########################

imag = CNN_images_feature()

################ Answer #########################
train_number = 140000 
validation_num = 146962 - train_number #146962
c = MAP_300TO5(0,1,train_number+validation_num)

### Train Golded Answer ###
ans_ABCDE_data_1 = open('dataset/pack/answer.train_sol','r')
ans_ABCDE_data = ans_ABCDE_data_1.readlines()
ans_ABCDE = []
for line in range(len(ans_ABCDE_data)):
    x = ans_ABCDE_data[line].split('\t')
    if x[0]!='img_id':
        xx = x[2].split('\n')[0]
        ans_ABCDE.append(xx)
ans_ABCDE_data_1.close()
print 'ans_ABCDE len',len(ans_ABCDE)

ans_300 = []
### Train Answer 300 ###
for i in range(len(ans_ABCDE)):
    ans_300.append(c.Get_300_Answer(i+1, ans_ABCDE[i]))
'''
################ Model ##########################
img_dim = 4096 #top layer of the VGG net
word_vec_dim = 300 #dimension of pre-trained word vectors

num_hidden_units = 2048,2048 #number of hidden units, a hyperparameter

num_hidden_units_lstm = 300
nb_classes = 300

image_model = Sequential()
image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))

language_model = Sequential()
language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))

model = Sequential()
model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
model.add(Dense(num_hidden_units[0], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_hidden_units[1], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')


filepath = 'Keras_0107_jason.txt'
model.fit([qv, imag], ans_300, nb_epoch=10, batch_size=16, show_accuracy=False, verbose=1)
model.save_weights(filepath)    
#model.load_weights(filepath)
print model.predict([qv[0],imag[0]], batch_size=1, verbose=0)
#model.fit_generator(gen(), samples_per_epoch=146962, nb_epoch=1,nb_worker=8,verbose=1, show_accuracy=True)
'''
########################################### TEST ############################################################
print "Start Test!"
################ Question #######################
ques_files = open('dataset/pack/question.test.trim', 'r'); 
ques = [l.split() for l in ques_files.read().splitlines()]
# qv = numpy.array((len(ques), max_len), dtype=float32)
qv = []
for x in ques:
    m = []
    for g in x:
        if g in dic:
            m.append(dic[g])
    t = len(m)
    for _ in range(20 - t): # max_len = 20
        m.append(numpy.zeros(300, dtype=numpy.float32))
    qv.append(numpy.vstack(m))
#qv = numpy.stack(qv)
qv = numpy.dstack(qv).transpose(2,0,1)
max_len = max([len(l) for l in qv])
#for x in qv:
    #for c in x:
     #   print(c.shape)
ques_files.close()

################ Image ###########################
imag = []
imag = CNN_images_feature(1)

############# choice ###############
choices_data = open('dataset/test_choice_vec','r')
ans_A_test = [] 
ans_B_test= [] 
ans_C_test = [] 
ans_D_test = [] 
ans_E_test = [] 
choices = choices_data.readlines()
i = 0
while (i < len(choices) ):
    x = choices[i].split(' ')[0:302]
    ix = [np.float32(ii) for ii in x[2:302]]
    if i%5 == 0 :
        ans_A_test.append(ix)
    elif i%5 == 1 :
        ans_B_test.append(ix)
    elif i%5 == 2 :
        ans_C_test.append(ix)
    elif i%5 == 3 :
        ans_D_test.append(ix)
    elif i%5 == 4 :
        ans_E_test.append(ix)
    i = i +1

choices_1 = open('dataset/pack/choices.test','r')
ABCDE_ans_A = [] 
ABCDE_ans_B = [] 
ABCDE_ans_C = [] 
ABCDE_ans_D = [] 
ABCDE_ans_E = [] 
name = []
not_first = 0
for line in choices_1:
    xx = line.split('\t')
    x = xx[2].split('  ')
    if not_first==1:
        ABCDE_ans_A.append(x[0].split('(A)')[1])
        ABCDE_ans_B.append(x[1].split('(B)')[1])
        ABCDE_ans_C.append(x[2].split('(C)')[1])
        ABCDE_ans_D.append(x[3].split('(D)')[1])
        ABCDE_ans_E.append(x[4].split('(E)')[1])
        name.append(xx[1])
    not_first = 1
        
haha = 0
for line in choices_1:
    haha = haha+1
print 'haha',haha

################ Model ##########################
img_dim = 4096 #top layer of the VGG net
word_vec_dim = 300 #dimension of pre-trained word vectors

num_hidden_units = 2048,2048 #number of hidden units, a hyperparameter

num_hidden_units_lstm = 300
nb_classes = 300

image_model = Sequential()
image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))

language_model = Sequential()
language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))

model = Sequential()
model.add(Merge([language_model, image_model], mode='concat', concat_axis=1))
model.add(Dense(num_hidden_units[0], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_hidden_units[1], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('tanh'))

model.compile(loss='mean_squared_error', optimizer='rmsprop')


filepath = 'Keras_0107_jason.txt'
#model.fit([qv, imag], ans_300, nb_epoch=1, batch_size=16, show_accuracy=False, verbose=1)
#model.save_weights(filepath)    
model.load_weights(filepath)

print "Start predict."
AA = model.predict([qv,imag] , batch_size=1, verbose=0)
print "End predict."

test_ans = open('LSTM_DNN_0107.csv','w')
test_ans.write('q_id,ans\n')
for i in range( len(qv) ):
    score = []
    for ii in range(5):
        if ii == 0:
            score.append(sum(AA[i]*ans_A_test[i]))
        elif ii ==1:
            score.append(sum(AA[i]*ans_B_test[i]))
        elif ii ==2:
            score.append(sum(AA[i]*ans_C_test[i]))
        elif ii==3:
            score.append(sum(AA[i]*ans_D_test[i]))
        elif ii==4:
            score.append(sum(AA[i]*ans_E_test[i]))
    case = score.index(max(score))

    answer_to_choose = [1,1,1,1,1] 
    candidates = [ABCDE_ans_A[i],ABCDE_ans_B[i],ABCDE_ans_C[i],ABCDE_ans_D[i],ABCDE_ans_E[i]]
    for iii in range (5): # Check for the same answer
        for ii in range (5):
            if ii != iii:
                if(candidates[iii] == candidates[ii]):
                    answer_to_choose[iii]=0
                    answer_to_choose[ii]=0

    while( answer_to_choose[case] == 0 ):
        if answer_to_choose == [0,0,0,0,0]:
            print 'id' , i
            print 'candidates',candidates
            break;
        score[case] = -1000000
        case = score.index(max(score))
        #print "GET YOU"
    test_ans.write(name[i])
    test_ans.write(',')
    if case == 0 :
        word = 'A'
    elif case == 1:
        word = 'B'
    elif case == 2:
        word = 'C'
    elif case == 3:
        word = 'D'
    elif case == 4:
        word = 'E'   
    #if Is_many[i]==1:
    #    word = Model_ans[i]  
    test_ans.write(word)
    if i!=len(qv)-1:
        test_ans.write('\n')
'''
