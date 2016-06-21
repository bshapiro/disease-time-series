config = dict(
    project_root='/Users/benj/Documents/Research/Projects/disease-time-series/',
    view_pca_plots=False,
    transpose_data=False,
    run_enrichment=True,
    run_clustering=True,
    gsea_file_prefix='_GO-BP_',
    trajectories='cleaned'
)


param = dict(
    dataset=None,
    representation='pca',
    components=3,
    kmeans_clusters=71,
    time_transform=False,
    clean_components=3,
    log_transform=True,
    clean_data=True,
    cca_reg=0.00001,
    gsea_fdr_thresh=0.3,
    gsea_p_value_thresh=0.05
)


def get_org_params():
    org_params = ['representation', 'clean_components']
    if param['representation'] == 'cca':
        org_params.append('cca_reg')
    return org_params


#TODO: write included config params for string gen
