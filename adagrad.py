def adagrad(x_data,y_data,epochs,batch_size=4,alpha=0.1,gamma=0.1,epsilon=1e-6):
  y_data = y_data.reshape((-1,1))
  theta = np.random.random(size=(x_data.shape[1]+1,1))
  theta_log = []
  loss =[]
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

      vt = vt + np.square((1/b)*(x_batch.T@(h-y_batch)))
      theta = theta - alpha*(1/(np.sqrt(vt)+epsilon)) * (1/b)*(x_batch.T@(h-y_batch))
      theta_log.append(theta)
  loss = np.array(loss)
  theta_log = np.array(theta_log)
  return theta ,loss,theta_log