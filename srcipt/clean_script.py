from util.clean_dataset import clean_dataset_by_clean_record

if __name__ == '__main__':
    CLEAN_RECORD = 'G:/workspace_plane2DDL/clean_record.xlsx'
    DATASET_ROOT = r'G:\workspace_plane2DDL\testData\anno_shp_with_normal_diffusion'
    clean_dataset_by_clean_record(DATASET_ROOT,CLEAN_RECORD)

