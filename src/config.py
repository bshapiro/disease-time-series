config = dict(
    project_root='/Users/benj/Documents/Research/Projects/disease-time-series/',
    view_pca_plots=False,
    transpose_data=False,
    run_enrichment=True,
    run_clustering=True,
    gsea_file_prefix='_GO-BP_',
)


param = dict(
    dataset=None,
    representation='cca',
    components=3,
    kmeans_clusters=71,
    time_transform=False,
    clean_components=3,
    clean_data=True,
    cca_reg=0.00001,
    gsea_fdr_thresh=0.3,
    gsea_p_value_thresh=0.05
)
