class LSTM:
    # LSTM cell (input, output, amount of recurrence, learning rate)
    def __init__ (self, xs, ys, rl, lr):
        #input is word length x word length
        self.x = np.zeros(xs+ys)
        #input size is word length + word length
        self.xs = xs + ys
        #output 
        self.y = np.zeros(ys)
        #output size
        self.ys = ys
        #cell state intialized as size of prediction
        self.cs = np.zeros(ys)
        #how often to perform recurrence
        self.rl = rl
        #balance the rate of training (learning rate)
        self.lr = lr
        #init weight matrices for our gates
        #forget gate
        self.f = np.random.random((ys, xs+ys))
        #input gate
        self.i = np.random.random((ys, xs+ys))
        #cell state
        self.c = np.random.random((ys, xs+ys))
        #output gate
        self.o = np.random.random((ys, xs+ys))
        #forget gate gradient
        self.Gf = np.zeros_like(self.f)
        #input gate gradient
        self.Gi = np.zeros_like(self.i)
        #cell state gradient
        self.Gc = np.zeros_like(self.c)
        #output gate gradient
        self.Go = np.zeros_like(self.o)
    
    #activation function to activate our forward prop, just like in any type of neural network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    #derivative of sigmoid to help computes gradients
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    #tanh! another activation function, often used in LSTM cells
    #Having stronger gradients: since data is centered around 0, 
    #the derivatives are higher. To see this, calculate the derivative 
    #of the tanh function and notice that input values are in the range [0,1].
    def tangent(self, x):
        return np.tanh(x)
    
    #derivative for computing gradients
    def dtangent(self, x):
        return 1 - np.tanh(x)**2
    
    #lets compute a series of matrix multiplications to convert our input into our output
    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o
    
   
    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        #error = error + hidden state derivative. clip the value between -6 and 6.
        e = np.clip(e + dfhs, -6, 6)
        #multiply error by activated cell state to compute output derivative
        do = self.tangent(self.cs) * e
        #output update = (output deriv * activated output) * input
        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        #derivative of cell state = error * output * deriv of cell state + deriv cell
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        #deriv of cell = deriv cell state * input
        dc = dcs * i
        #cell update = deriv cell * activated cell * input
        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        #deriv of input = deriv cell state * cell
        di = dcs * c
        #input update = (deriv input * activated input) * input
        iu = np.dot(np.atleast_2d(di * self.dsigmoid(i)).T, np.atleast_2d(self.x))
        #deriv forget = deriv cell state * all cell states
        df = dcs * pcs
        #forget update = (deriv forget * deriv forget) * input
        fu = np.dot(np.atleast_2d(df * self.dsigmoid(f)).T, np.atleast_2d(self.x))
        #deriv cell state = deriv cell state * forget
        dpcs = dcs * f
        #deriv hidden state = (deriv cell * cell) * output + deriv output * output * output deriv input * input * output + deriv forget
        #* forget * output
        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df, self.f)[:self.ys] 
        #return update gradinets for forget, input, cell, output, cell state, hidden state
        return fu, iu, cu, ou, dpcs, dphs
            
    def update(self, fu, iu, cu, ou):
        #update forget, input, cell, and output gradients
        self.Gf = 0.9 * self.Gf + 0.1 * fu**2 
        self.Gi = 0.9 * self.Gi + 0.1 * iu**2   
        self.Gc = 0.9 * self.Gc + 0.1 * cu**2   
        self.Go = 0.9 * self.Go + 0.1 * ou**2   
        
        #update our gates using our gradients
        self.f -= self.lr/np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr/np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr/np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr/np.sqrt(self.Go + 1e-8) * ou
        return
    
    
    
    
    
    
    
    class_labels = pd.read_csv('labels.csv',encoding='utf-8')
weighted_tfidf_score = pd.read_csv('tfidf_scores.csv',encoding='utf-8')
sentiment_scores = pd.read_csv('sentiment_scores.csv',encoding='utf-8')
dependency_features = pd.read_csv('dependency_features.csv',encoding='utf-8')
char_bigrams = pd.read_csv('char_bigram_features.csv',encoding='utf-8')
word_bigrams = pd.read_csv('word_bigram_features.csv',encoding='utf-8')
tfidf_sparse_matrix = pd.read_csv('tfidf_features.csv',encoding='utf-8')

#merge all feature data sets based on 'index' column sentiment_scores, dependency_features, char_bigrams, word_bigrams
df_list=[class_labels, weighted_tfidf_score,sentiment_scores, dependency_features, char_bigrams, word_bigrams, tfidf_sparse_matrix]
master = df_list[0]
for df in df_list[1:]:
    master = master.merge(df, on='index')

master.columns.values
#ignore first two columns (index and tweet)

y=master.iloc[:,2] #class labels
X=master.iloc[:,3:] #all features


#create train and test sets: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##########################################################################################
#NOW WE CAN START MODELING
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
from sklearn import model_selection
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

#Create a base training set to benchmark our performance (train set with hatespeech dictionary weighted tif-df score as only feature)
x_base = pd.DataFrame(X_train.loc[:,'weighted_TFIDF_scores'])
x_base_test = pd.DataFrame(X_test.loc[:,'weighted_TFIDF_scores'])

# created scaled version of training and test data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

#initialize models
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=.025)
xgb = XGBClassifier(learning_rate=.025, max_features=100)
mlp = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(80,40,40,10), activation='relu', random_state=1,learning_rate='adaptive', alpha=1e-6)
rf= RandomForestClassifier(n_estimators=100, max_features=500)
# 80,50,50,20

#asses model performances using 5-fold cross validation and f1-score micro aveage as metric
print("baseline model f1-score = ", cross_val_score(lr,x_base, y_train,cv=5,scoring="roc_auc").mean()) #benchmark model: linear regression using just tfidf score (weighted with hate dict)
print("gb cross validation f1-score = ", cross_val_score(gb,x_base, y_train,cv=5,scoring="f1_micro").mean()) #gradient boost with just tf-df score
print("rf cross validation f1-score = ", cross_val_score(rf,X_train,y_train,cv=5,scoring="f1_micro").mean()) #random forest with full train set (all features)
print("xgb cross validation f1-score = ", cross_val_score(xgb,X_train,y_train,cv=5,scoring="f1_micro").mean()) #xgboost with full train set (all features)
print("mlp cross validation f1-score = ", cross_val_score(mlp,X_train,y_train,cv=5,scoring="f1_micro").mean())

#initialize ensembles
estimators=[]
estimators.append(('mlp', mlp))
estimators.append(('rf', rf))
estimators.append(('xgb', xgb))

#voting ensemlbe
ensemble = VotingClassifier(estimators, voting='soft',weights=[1,1,1])
ensemble.fit(X_train, y_train)
pred = ensemble.predict(X_test)
print ('fscore:{0:.3f}'.format(f1_score(y_test, pred, average='micro')))

#meta classifier ensemble
stack = StackingCVClassifier(classifiers=[mlp, xgb, rf], cv=2,meta_classifier=lr, use_probas=True)
stack.fit(X_train.values, y_train.values)
pred2=stack.predict(X_test.values)
print ('fscore:{0:.3f}'.format(f1_score(y_test, pred2, average='micro')))

from sklearn.metrics import confusion_matrix
confusion_lr = confusion_matrix(y_test, pred)
print(confusion_lr)
