from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from missingpy import MissForest
import torch
from tqdm import tqdm
import torch.nn.functional as F

 def auto_imput1(df_train_part_dirty):
 	imper = IterativeImputer(max_iter=10, random_state=0)
 	imp.fit(df_train_part_dirty)
 	df_train_part_clean = imp.transform(df_train_part_dirty)
 	return df_train_part_clean


def auto_imput2(df_train_part_dirty):
	imputer = MissForest()
	df_train_part_clean = imputer.fit_transform(df_train_part_dirty)
	return df_train_part_clean


def auto_imput3(df_train_part_dirty):

	def generator(new_x,m):
    inputs = torch.cat(dim = 1, tensors = [new_x,m]) 
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) 
    
    return G_prob


    def discriminator(new_x, h):
    inputs = torch.cat(dim = 1, tensors = [new_x,h]) 
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)


    def d_loss(M, New_X, H):
    G_sample = generator(New_X,M)
    Hat_New_X = New_X * M + G_sample * (1-M)

    D_prob = discriminator(Hat_New_X, H)
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
    return D_loss


	def g_loss(X, M, New_X, H):
	    G_sample = generator(New_X,M)
	    Hat_New_X = New_X * M + G_sample * (1-M)
	    D_prob = discriminator(Hat_New_X, H)
	    G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
	    MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

	    G_loss = G_loss1 + alpha * MSE_train_loss 

	    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
	    return G_loss, MSE_train_loss, MSE_test_loss
    
	def t_loss(X, M, New_X):
	    G_sample = generator(New_X,M)
	    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
	    return MSE_test_loss, G_sample

    
    return D_prob

	Dim = len(df_train_part_dirty)

	D_W1 = torch.tensor(xavier_init([Dim*2, Dim]),requires_grad=True, device="cuda")
	D_b1 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")

	D_W2 = torch.tensor(xavier_init([Dim, Dim]),requires_grad=True, device="cuda")
	D_b2 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")

	D_W3 = torch.tensor(xavier_init([Dim, Dim]),requires_grad=True, device="cuda")
	D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")

	theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


	G_W1 = torch.tensor(xavier_init([Dim*2, Dim]),requires_grad=True, device="cuda")     
    G_b1 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")

    G_W2 = torch.tensor(xavier_init([Dim, Dim]),requires_grad=True, device="cuda")
    G_b2 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")

    G_W3 = torch.tensor(xavier_init([Dim, Dim]),requires_grad=True, device="cuda")
    G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")


    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


    optimizer_D = torch.optim.Adam(params=theta_D)
	optimizer_G = torch.optim.Adam(params=theta_G)



	for it in tqdm(range(5000)):    
	    mb_idx = sample_idx(Train_No, mb_size)
	    X_mb = trainX[mb_idx,:]  
	    
	    Z_mb = sample_Z(mb_size, Dim) 
	    M_mb = trainM[mb_idx,:]  
	    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
	    H_mb = M_mb * H_mb1
	    
	    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
	    
	    if use_gpu is True:
	        X_mb = torch.tensor(X_mb, device="cuda")
	        M_mb = torch.tensor(M_mb, device="cuda")
	        H_mb = torch.tensor(H_mb, device="cuda")
	        New_X_mb = torch.tensor(New_X_mb, device="cuda")

	    
	    optimizer_D.zero_grad()
	    D_loss_curr = d_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
	    D_loss_curr.backward()
	    optimizer_D.step()
	    
	    optimizer_G.zero_grad()
	    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = g_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
	    G_loss_curr.backward()
	    optimizer_G.step()    

	    , Sample = t_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
	    df_train_part_clean = M_mb * X_mb + (1-M_mb) * Sample

	    return df_train_part_clean



def actclean(dirty_data, clean_data, test_data, full_data, indextuple, batchsize=50, total=10000):
	X = dirty_data[0][translate_indices(indextuple[0],indextuple[1]),:]
	y = dirty_data[1][translate_indices(indextuple[0],indextuple[1])]

	X_clean = clean_data[0]
	y_clean = clean_data[1]

	X_test = test_data[0]
	y_test = test_data[1]


	lset = set(indextuple[2])
	dirtyex = [i for i in indextuple[0]]
	cleanex = []

	total_labels = []

	topbatch = np.random.choice(range(0,len(dirtyex)), batchsize)
	examples_real = [dirtyex[j] for j in topbatch]
	examples_map = translate_indices(examples_real, indextuple[2])

	cleanex.extend(examples_map)
	for j in examples_real:
		dirtyex.remove(j)


	clf = SGDClassifier(loss="hinge", alpha=0.000001, n_iter=200, fit_intercept=True, warm_start=True)
	clf.fit(X_clean[cleanex,:],y_clean[cleanex])

	for i in range(50, total, batchsize):
		ypred = clf.predict(X_test)
		print classification_report(y_test, ypred)

		examples_real = np.random.choice(dirtyex, batchsize)
		examples_map = translate_indices(examples_real, indextuple[2])

		total_labels.extend([(r, (r in lset)) for r in examples_real])
		
		ec = error_classifier(total_labels, full_data)

		for j in examples_real:
			dirtyex.remove(j)
			

		dirtyex = ec_filter(dirtyex, full_data, ec)


		cleanex.extend(examples_map)

		clf.partial_fit(X_clean[cleanex,:],y_clean[cleanex])

		if len(dirtyex) < 50:
			break


def btclean(df_train_part_dirty):
	from bclean import bclean_model

	clean_m = bclean_model()
	df_train_part_clean = clean_m.transform(df_train_part_dirty)
	return df_train_part_clean