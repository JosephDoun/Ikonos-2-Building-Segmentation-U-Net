<!-- <style>
    ul {
        background-color: #111;
        padding: 30px;
    }
    img.dlwpt {
        float: right;
    }
    div.training {
        padding: 60px;
    }
    li {
        padding:30px;
    }
    img {
        /* margin:10px; */
        margin-left: 30px;
        border-style: solid;
    }
</style> -->

<body>
    <div class='title'>
        <h1 align='center'> A U-Net implementation for Building Segmentation on Ikonos-2 Satellite Images </h1>
        <h3 align='center'>(Work in Progress)</h3>
        <p align='center'>
            Project based on the original <a href="https://arxiv.org/abs/1505.04597">U-Net paper</a>
            by Olaf Ronneberger, Philipp Fischer and Thomas Brox (2015)
        </p>
    </div>
    <div class='sections'>
        <div class='data'>
            <h2>1. Data</h2>
            <ul>
                <li>Ikonos-2 Multispectral images consisted of a Blue, Green, Red, and Near-Infrared channel. Ikonos-2 images come at a Spatial Resolution of 0.8 meters and a Radiometric Resolution of 11 bits.</li>
                <li>Initial training phase includes samples from 7 sub-areas of an image of the greater Thessaloniki Region, Greece, taken in Spring. This phase aims to evaluate model performance overall, as well as its generalization capabilities on images acquired on different seasons, before the dataset can be expanded. <br>
                <div align='center'>
                    <img align='center' src='./imgdir/training_areas.png' alt='Training areas' style='float:center;'> <br> &copy DigitalGlobe
                </div>
                </li>
                <li>Sample areas were delineated in QGIS and samples were collected from industrial and urban environments equally. Further samples were taken from irregular background areas. Extracted rasters were processed further into normalized tiles, separated in positive and negative samples and stored in hdf5 format. About 1/6 of each sub-area was kept for validation.</li>
            </ul>
            <footer align='center'>The data was purchased and provided by the Aristotle University of Thessaloniki</footer>
        </div>
        <div style='border-width:1px;padding:5px' class='training'>
            <h2>2. Training Environment</h2>
            <div align='center' style='background-color:#A88;padding:15px' class='training'>
                <p align='left'>Training mainly followed the recommendations of Ronneberger et al. (2015).
                    Additional training ideas and methods, such as class balancing, <br> were adopted from
                    <a href='https://www.google.com/search?channel=fs&client=ubuntu&q=deep+learning+with+pytorch'>Deep Learning with PyTorch</a> by Eli Stevens, Luca Antiga and Thomas Viehmann (2020)
                </p> 
                <div>
                    <a href='https://www.google.com/search?channel=fs&client=ubuntu&q=deep+learning+with+pytorch'>
                        <img src='https://images.manning.com/book/3/8e5d003-09e3-430e-a5a3-f42ee1cafb5f/Stevens-DLPy-HI.png' width=5% height=5%
                        alt='Deep Learning with PyTorch' class='dlwpt' style='float: right;'>
                    </a>
                </div>
            </div>
            <ul>
                <li>
                    Adam was used with a high momentum, as recommended in Ronneberger et al. 2015. Training samples, however, were sized 64 by 64 pixels, to allow class balancing flexibility. Validation samples remained large at 512 by 512 pixels.
                </li>
                <li>
                    Augmentation includes affine transformations (Translation, Rotation, Scaling and Shear), noise, brightness and contrast adjustments, as well as elastic deformations. Elastic deformations proved to be as crucial to training as claimed in the U-Net paper. Coarse smaller kernel deformations of mild intensity appear to be performing better, contrary to larger kernels with higher intensity.
                </li>
                <li>
                    Elastic deformation was implemented according to Microsoft paper <a href='https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf'>Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis</a>. 
                </li>
                <li>The implementation of symmetric dropout layers that peak at the bottom of the U-Net were found crucial.
                </li>
            </ul>
        </div>
    </div>
</body>
<footer>
Iosif Doundoulakis <br>
iosif.doundoulakis@outlook.com
</footer>
