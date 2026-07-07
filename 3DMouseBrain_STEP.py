from STEP import run
import glob
import os

for i in range(1, 36):
    tissue = str(i) if i >= 10 else '0' + str(i)
    print(tissue)
    Img_Data = glob.glob(f'/data/hzq/idea/Mouse_brain_3D/GSE147747_RAW/*_HE_{tissue}A.jpg')[0]
    Json_Data = glob.glob(f'/data/hzq/idea/Mouse_brain_3D/GSE147747_RAW/*_HE_{tissue}A.geojson')[0]
    CLAM_Data = glob.glob(f'/data/hzq/idea/Mouse_brain_3D/3D/patches/*_HE_{tissue}A.h5')[0]
    SC_Data = '/data/hzq/idea/Mouse_brain_3D/E-MTAB-11115/sc.h5ad'
    cell_class_column = 'annotation_1'
    out_dir = '/data/hzq/idea/Mouse_brain_3D/output'
    
    run.ExtractFeatures(    
        tissue=tissue,
        out_dir=out_dir,
        ST_Data=f'/data/hzq/idea/Mouse_brain_3D/mouse_brain_st/{tissue}.h5ad',
        Img_Data=Img_Data,
        CLAM_Data=CLAM_Data,
        Json_Data=Json_Data)
    
    run.CellIdentification(
        tissue=tissue,
        out_dir=out_dir,
        ST_Data=os.path.join(out_dir, tissue, 'sp_adata_ef.h5ad'),
        SC_Data=SC_Data,
        cell_class_column=cell_class_column,
        device='cuda:3')

    run.GeneEnhancement(
        tissue=tissue,
        out_dir=out_dir,
        ST_Data=os.path.join(out_dir, tissue, 'sp_adata_ef.h5ad'),
        SC_Data=SC_Data,
        cell_class_column=cell_class_column
        )