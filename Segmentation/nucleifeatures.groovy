/**
 * This script provides a general template for nucleus detection using StarDist in QuPath.
 * This example assumes you have an RGB color image, e.g. a brightfield H&E slide.
 * 
 * If you use this in published work, please remember to cite *both*:
 *  - the original StarDist paper (https://doi.org/10.48550/arXiv.1806.03535)
 *  - the original QuPath paper (https://doi.org/10.1038/s41598-017-17204-5)
 *  
 * There are lots of options to customize the detection - this script shows some 
 * of the main ones. Check out other scripts and the QuPath docs for more info.
 */

import qupath.ext.stardist.StarDist2D
import qupath.lib.scripting.QP
setImageType('BRIGHTFIELD_H_E');
setColorDeconvolutionStains('{"Name" : "H&E default", "Stain 1" : "Hematoxylin", "Values 1" : "0.60968 0.65246 0.4501 ", "Stain 2" : "Eosin", "Values 2" : "0.21306 0.87722 0.43022 ", "Background" : " 243 243 243 "}');
// IMPORTANT! Replace this with the path to your StarDist model
// that takes 3 channel RGB as input (e.g. he_heavy_augment.pb)
// You can find some at https://github.com/qupath/models
// (Check credit & reuse info before downloading)
def modelPath = "/Users/huzheqi/Downloads/he_heavy_augment.pb" // Change to your path

// Customize how the StarDist detection should be applied
// Here some reasonable default options are specified
def cell_expansion_factor = 3.0
def cellConstrainScale = 1.0
def stardist = StarDist2D.builder(modelPath)
        .threshold(0.5)              // Probability (detection) threshold
        .normalizePercentiles(1, 99) // Percentile normalization
        .pixelSize(getCurrentServer().getPixelCalibration().getAveragedPixelSize() * 2) // .pixelSize(0.5)              // Resolution for detection
        .cellExpansion(cell_expansion_factor)          // Approximate cells based upon nucleus expansion
        .cellConstrainScale(cellConstrainScale)     // Constrain cell expansion using nucleus size
        .measureShape()              // Add shape measurements
        .measureIntensity()          // Add cell measurements (in all compartments)
        .includeProbability(true)    // Add probability as a measurement (enables later filtering)
        .nThreads(10)
        .build()
	 
// Define which objects will be used as the 'parents' for detection
// Use QP.getAnnotationObjects() if you want to use all annotations, rather than selected objects
def pathObjects = QP.getSelectedObjects()

// Run detection for the selected objects
def imageData = QP.getCurrentImageData()
if (pathObjects.isEmpty()) {
    QP.getLogger().error("No parent objects are selected!")
    return
}
def server = imageData.getServer()

// get dimensions of slide
minX = 0
minY = 0
maxX = server.getWidth()
maxY = server.getHeight()

stardist.detectObjects(imageData, pathObjects)
stardist.close() // This can help clean up & regain memory
runObjectClassifier("/Users/huzheqi/Downloads/ANN_StardistSeg3.0CellExp1.0CellConstraint_AllFeatures_LymphClassifier.json") // Change to your path
println('Done!')


