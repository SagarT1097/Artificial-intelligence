{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8c8cf948",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras import datasets, layers, models\n",
    "from keras.models import Sequential\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from keras.datasets import cifar10\n",
    "from keras.utils import np_utils\n",
    "from keras import backend as K\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "911ce25b",
   "metadata": {},
   "outputs": [],
   "source": [
    "class CIFAR:\n",
    "    def __init__(self,seed=0):\n",
    "        # Get and split data\n",
    "        data = self.__getData(seed)\n",
    "        self.x_train_raw=data[0][0]\n",
    "        self.y_train_raw=data[0][1]\n",
    "        self.x_valid_raw=data[1][0]\n",
    "        self.y_valid_raw=data[1][1]\n",
    "        self.x_test_raw=data[2][0]\n",
    "        self.y_test_raw=data[2][1]\n",
    "        # Record input/output dimensions\n",
    "        self.num_classes=10\n",
    "        self.input_dim=self.x_train_raw.shape[1:]\n",
    "         # Convert data\n",
    "        self.y_train = np_utils.to_categorical(self.y_train_raw, self.num_classes)\n",
    "        self.y_valid = np_utils.to_categorical(self.y_valid_raw, self.num_classes)\n",
    "        self.y_test = np_utils.to_categorical(self.y_test_raw, self.num_classes)\n",
    "        self.x_train = self.x_train_raw.astype('float32')\n",
    "        self.x_valid = self.x_valid_raw.astype('float32')\n",
    "        self.x_test = self.x_test_raw.astype('float32')\n",
    "        self.x_train  /= 255\n",
    "        self.x_valid  /= 255\n",
    "        self.x_test /= 255\n",
    "        # Class names\n",
    "        self.class_names=['airplane','automobile','bird','cat','deer',\n",
    "               'dog','frog','horse','ship','truck']\n",
    "\n",
    "    def __getData (self,seed=0):\n",
    "        (x_train, y_train), (x_test, y_test) = cifar10.load_data()\n",
    "        return self.__shuffleData(x_train,y_train,x_test,y_test,seed)\n",
    "\n",
    "    def __shuffleData (self,x_train,y_train,x_test,y_test,seed=0):\n",
    "        tr_perc=.75\n",
    "        va_perc=.15\n",
    "        x=np.concatenate((x_train,x_test))\n",
    "        y=np.concatenate((y_train,y_test))\n",
    "        np.random.seed(seed)\n",
    "        np.random.shuffle(x)\n",
    "        np.random.seed(seed)\n",
    "        np.random.shuffle(y)\n",
    "        indices = np.random.permutation(len(x))\n",
    "        tr=round(len(x)*tr_perc)\n",
    "        va=round(len(x)*va_perc)\n",
    "        self.tr_indices=indices[0:tr]\n",
    "        self.va_indices=indices[tr:(tr+va)]\n",
    "        self.te_indices=indices[(tr+va):len(x)]\n",
    "        x_tr=x[self.tr_indices,]\n",
    "        x_va=x[self.va_indices,]\n",
    "        x_te=x[self.te_indices,]\n",
    "        y_tr=y[self.tr_indices,]\n",
    "        y_va=y[self.va_indices,]\n",
    "        y_te=y[self.te_indices,]\n",
    "        return ((x_tr,y_tr),(x_va,y_va),(x_te,y_te))\n",
    "\n",
    "    # Print 25 random figures from the validation data\n",
    "    def showImages(self):\n",
    "        images=self.x_valid_raw\n",
    "        labels=self.y_valid_raw\n",
    "        class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',\n",
    "               'dog', 'frog', 'horse', 'ship', 'truck']\n",
    "        plt.figure(figsize=(10,10))\n",
    "        indices=np.random.randint(0,images.shape[0],25)\n",
    "        for i in range(25):\n",
    "            plt.subplot(5,5,i+1)\n",
    "            plt.xticks([])\n",
    "            plt.yticks([])\n",
    "            plt.grid(False)\n",
    "            plt.imshow(images[indices[i]], cmap=plt.cm.binary)\n",
    "            # The CIFAR labels happen to be arrays,\n",
    "            # which is why we need the extra index\n",
    "            plt.xlabel(class_names[labels[indices[i]][0]])\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "42078ed1",
   "metadata": {},
   "outputs": [],
   "source": [
    "dataSet = CIFAR()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "c2b24d8e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def myCNNModel():\n",
    "    model = Sequential([\n",
    "        layers.Conv2D(32, 3, padding='same', activation='relu', ),\n",
    "        layers.Dropout(0.5),\n",
    "        layers.Conv2D(32, 3, padding='same', activation='relu', ),\n",
    "        layers.MaxPooling2D(),\n",
    "        layers.Dropout(0.25),\n",
    "        layers.Conv2D(64, 3, padding='same', activation='relu', ),\n",
    "        layers.Dropout(0.5),\n",
    "        layers.Conv2D(64, 3, padding='same', activation='relu', ),\n",
    "        layers.MaxPooling2D(),\n",
    "        layers.Dropout(0.25),\n",
    "        layers.Flatten(),\n",
    "        layers.Dense(128,activation='relu'),\n",
    "        layers.Softmax(),\n",
    "        ])\n",
    "    return model\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "aef28743",
   "metadata": {},
   "outputs": [],
   "source": [
    "def myGetModel(dataSet):\n",
    "    model = myCNNModel()\n",
    "    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),\n",
    "              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "dd475717",
   "metadata": {},
   "outputs": [],
   "source": [
    "def myFitModel(myModel, dataSet):\n",
    "    myModel = myCNNModel()\n",
    "    myModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),\n",
    "                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n",
    "                  metrics=['accuracy'])\n",
    "    history = myModel.fit(dataSet.x_train, dataSet.y_train, epochs=1,\n",
    "                        validation_data=(dataSet.x_valid, dataSet.y_valid),\n",
    "                        batch_size=64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f990e178",
   "metadata": {},
   "outputs": [],
   "source": [
    "def runImageClassification(getModel=myGetModel,fitModel=myFitModel,seed=7):\n",
    "    # Fetch data. You may need to be connected to the internet the first time this is done.\n",
    "    # After the first time, it should be available in your system. On the off chance this\n",
    "    # is not the case on your system and you find yourself repeatedly downloading the data,\n",
    "    # you should change this code so you can load the data once and pass it to this function.\n",
    "    print(\"Preparing data...\")\n",
    "    data=CIFAR(seed)\n",
    "\n",
    "    # Create model\n",
    "    print(\"Creating model...\")\n",
    "    model=getModel(data)\n",
    "\n",
    "    # Fit model\n",
    "    print(\"Fitting model...\")\n",
    "    model=fitModel(model,data)\n",
    "\n",
    "    # Evaluate on test data\n",
    "    print(\"Evaluating model...\")\n",
    "    score = model.evaluate(data.x_test, data.y_test, verbose=0)\n",
    "    print('Test accuracy:', score[1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7c72c781",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preparing data...\n",
      "Creating model...\n",
      "Fitting model...\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "Protocols cannot be instantiated",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_236/4038873675.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mrunImageClassification\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmyGetModel\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mmyFitModel\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mseed\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m7\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_236/438058407.py\u001b[0m in \u001b[0;36mrunImageClassification\u001b[1;34m(getModel, fitModel, seed)\u001b[0m\n\u001b[0;32m     13\u001b[0m     \u001b[1;31m# Fit model\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     14\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Fitting model...\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 15\u001b[1;33m     \u001b[0mmodel\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mfitModel\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     16\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     17\u001b[0m     \u001b[1;31m# Evaluate on test data\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_236/3852442008.py\u001b[0m in \u001b[0;36mmyFitModel\u001b[1;34m(myModel, dataSet)\u001b[0m\n\u001b[0;32m      4\u001b[0m                   \u001b[0mloss\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mkeras\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlosses\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mSparseCategoricalCrossentropy\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfrom_logits\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m                   metrics=['accuracy'])\n\u001b[1;32m----> 6\u001b[1;33m     history = myModel.fit(dataSet.x_train, dataSet.y_train, epochs=1,\n\u001b[0m\u001b[0;32m      7\u001b[0m                         \u001b[0mvalidation_data\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mdataSet\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mx_valid\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdataSet\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0my_valid\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m                         batch_size=64)\n",
      "\u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\keras\\utils\\traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     65\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m  \u001b[1;31m# pylint: disable=broad-except\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     66\u001b[0m       \u001b[0mfiltered_tb\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 67\u001b[1;33m       \u001b[1;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfiltered_tb\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     68\u001b[0m     \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     69\u001b[0m       \u001b[1;32mdel\u001b[0m \u001b[0mfiltered_tb\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Local\\Programs\\Python\\Python39\\lib\\typing.py\u001b[0m in \u001b[0;36m_no_init\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1081\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1082\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0m_no_init\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1083\u001b[1;33m     \u001b[1;32mraise\u001b[0m \u001b[0mTypeError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'Protocols cannot be instantiated'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1084\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1085\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mTypeError\u001b[0m: Protocols cannot be instantiated"
     ]
    }
   ],
   "source": [
    "runImageClassification(myGetModel,myFitModel,seed=7)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d2e2bea7",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
