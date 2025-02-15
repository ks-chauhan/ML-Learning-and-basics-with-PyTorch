{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPZWHVfII9FF4RG60pIlysj",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ks-chauhan/ML-Learning-and-basics-with-PyTorch/blob/main/helper_functions.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "HOlF1Ppy7CbH"
      },
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "def plot_decision_boundary(model:torch.nn.Module,X:torch.tensor,y:torch.tensor):\n",
        "  X,y=X.to(\"cpu\"),y.to(\"cpu\")\n",
        "  #\n",
        "  x_min,x_max=X[:,0].min(),X[:,0].max()\n",
        "  y_min,y_max=X[:,1].min(),X[:,1].max()\n",
        "  xx,yy=np.meshgrid(np.linspace(x_min,x_max,101),np.linspace(y_min,y_max,101))\n",
        "  #\n",
        "  X_pred_on=torch.from_numpy(np.column_stack(xx.ravel(),yy.ravel())).float()\n",
        "  #\n",
        "  with torch.inference_mode():\n",
        "    pred=model(X_pred_on)\n",
        "  #\n",
        "  if(len(torch.unique(y))>2):\n",
        "    pred=torch.softmax(pred).argmax(dim=1)\n",
        "  else:\n",
        "    pred=torch.sigmoid(pred).round()\n",
        "  #\n",
        "  pred=pred.reshape(xx.shape).detach().numpy()\n",
        "  plt.contour(xx,yy,pred,cmap=plt.cm.RdYlBU,alpha=0.7)\n",
        "  plt.scatter(X[:,0],Y[:,1],y,cmap=plt.cm.RdYlBU)\n",
        "  plt.xlim(xx.min(),xx.max())\n",
        "  plt.ylim(yy.min(),yy.max())"
      ],
      "metadata": {
        "id": "cmys8-NnhXlD"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "NuP0DevQ_8Wh"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}