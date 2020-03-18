from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

print('Getting MNIST Data..')
mnist = fetch_openml('mnist_784',version=1,cache=True)
print('MNIST Data downloaded!')

images = mnist.data
labels = mnist.target

images = normalize(images,norm='l2')

images_train,images_test,labels_train,labels_test = train_test_split(images,labels,test_size=0.25,random_state=17)

nn = MLPClassifier(hidden_layer_sizes=(100),max_iter=20,solver='sgd',learning_rate_init=0.001,verbose=True)

print('NN Training started..')
nn.fit(images_train,labels_train)
print('NN Training completed!')

print('Network Performance: %f' % nn.score(images_test,labels_test))
