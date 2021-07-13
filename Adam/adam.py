def adam(x_data,y_data,epochs,batch_size=4,alpha=0.001,beta_1=0.9,beta_2=0.99,epsilon=1e-8):
  y_data = y_data.reshape((-1,1))
  theta = np.random.random(size=(x_data.shape[1]+1,1))
  theta_log = []
  loss =[]
  t = 1 
  mt = np.zeros((x_data.shape[1]+1,1))
  vt = np.zeros((x_data.shape[1]+1,1))
  m = len(x_data)
  b = batch_size
  steps = m//b if (m%b)==0 else (m//b)+1
  for _ in range(epochs):
    for step in range(steps):
      x_batch = x_data[step*b:(step+1)*b]
      y_batch = y_data[step*b:(step+1)*b]
      x_batch = np.concatenate((np.ones((x_batch.shape[0],1)),x_batch),axis=1)
      h = x_batch @ theta
      
      loss.append((1/(2*b))*np.sum(np.square(h-y_batch)))
      mt = beta_1 * mt + (1 - beta_1) * (1/b)*(x_batch.T@(h-y_batch))
      vt = beta_2 * vt + (1 - beta_2) * np.square((1/b)*(x_batch.T@(h-y_batch)))
      mt = mt /(1-beta_1**t)
      vt = vt /(1-beta_2**t)
      theta = theta - alpha*(1/(np.sqrt(vt)+epsilon)) * mt 

      t+=1
      theta_log.append(theta)
  loss = np.array(loss)
  theta_log = np.array(theta_log)
  return theta ,loss,theta_log