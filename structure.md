# Structure du projet

```
└── 📁Devoir
    └── .gitignore
    └── 📁data
        └── 📁MNIST
            └── 📁raw
                └── t10k-images-idx3-ubyte
                └── t10k-images-idx3-ubyte.gz
                └── t10k-labels-idx1-ubyte
                └── t10k-labels-idx1-ubyte.gz
                └── train-images-idx3-ubyte
                └── train-images-idx3-ubyte.gz
                └── train-labels-idx1-ubyte
                └── train-labels-idx1-ubyte.gz
    └── 📁models
        └── .gitkeep
    └── 📁notebooks # Projets jouets
        └── .gitkeep
        └── 📁CNN_demo
            └── 📁data
                └── 📁MNIST
                    └── 📁raw
                        └── t10k-images-idx3-ubyte
                        └── t10k-images-idx3-ubyte.gz
                        └── t10k-labels-idx1-ubyte
                        └── t10k-labels-idx1-ubyte.gz
                        └── train-images-idx3-ubyte
                        └── train-images-idx3-ubyte.gz
                        └── train-labels-idx1-ubyte
                        └── train-labels-idx1-ubyte.gz
            └── device_availability.py
            └── 📁output
                └── model.pth
                └── plot.png
                └── plot1.png
                └── plot2.png
                └── plot3.png
            └── predict.py
            └── 📁pyimagesearch
                └── ConvolutionalLayer.py
                └── __init__.py
                └── 📁__pycache__
                    └── ConvolutionalLayer.cpython-311.pyc
                    └── __init__.cpython-311.pyc
            └── train.py
        └── processus.ipynb
        └── reading_mnist.ipynb
        └── tutorial_quanvolution.ipynb
    └── Readme.md
    └── 📁references
        └── .gitkeep
        └── QML_for_image_classification.pdf
        └── Quanv_GTI755.png
    └── 📁results
        └── .gitkeep
        └── 📁01-04-2024@22-25-20
            └── model.pth
            └── plot.png
        └── 📁01-04-2024@22-54-25
        └── 📁01-04-2024@23-18-33
        └── 📁14-03-2024@13-51-10
        └── 📁14-03-2024@13-52-26
        └── 📁2024-04-04@17-41-10
        └── 📁2024-04-08@11-21-37
        └── 📁2024-04-08@12-26-01
        └── 📁2024-04-08@12-56-36
        └── 📁2024-04-08@19-32-08
        └── 📁2024-04-08@19-46-27
        └── 📁2024-04-08@20-17-27
        └── 📁2024-04-09@09-44-45
        └── 📁2024-04-09@10-15-56
        └── 📁2024-04-09@11-08-19
        └── 📁2024-04-09@11-47-14
        └── 📁2024-04-09@12-24-36
        └── 📁2024-04-09@15-47-00
        └── 📁2024-04-09@16-11-18
        └── 📁2024-04-09@16-51-23

    └── 📁src
        └── constants.py
        └── 📁layers
            └── .gitkeep
            └── ConvolutionalLayer.py
            └── 📁old
                └── QuanvolutionalLayer.py
            └── QuanvolutionLayer.py
            └── VQCLayer.py
            └── __init__.py
            └── 📁__pycache__
                └── QuanvolutionLayer.cpython-311.pyc
                └── VQCLayer.cpython-311.pyc
                └── __init__.cpython-311.pyc
        └── 📁models
            └── .gitkeep
            └── HQNN_Parallel.py
            └── HQNN_Quanv.py
            └── LeNet.py
            └── __init__.py
            └── 📁__pycache__
                └── HQNN_Parallel.cpython-311.pyc
                └── HQNN_Quanv.cpython-311.pyc
                └── LeNet.cpython-311.pyc
                └── __init__.cpython-311.pyc
        └── predict.py
        └── train.py
        └── utils.py
        └── visualisation.py
        └── __init__.py
```