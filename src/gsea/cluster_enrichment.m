folder_dir = '../../data/gsea_output/';
folders = dir(strcat(folder_dir, 'cleaned*'));
isub = [folders(:).isdir]; %# returns logical vector
nameFolds = {folders(isub).name}';
genes = load(strcat(folder_dir, 'myeloma_genes.mat'));
for i=1:numel(nameFolds)
    pca_file = dir(strcat(folder_dir, nameFolds{i}, '/myeloma_pca*.mat'));
    pca_filename = pca_file.name;
    clusters = load(strcat(folder_dir, nameFolds{i}, '/', pca_filename));

    cluster_names = fieldnames(clusters);
    for j=1:numel(cluster_names)
        cluster = clusters.(cluster_names{j});
        [enrich_data, enrich_genes] = geneset_enrichment(genes.genes, cluster, 'GO-BP', strcat(folder_dir, nameFolds{i}, '/svd_', char(cluster_names{j})), 20);
    end
    
    cca_file = dir(strcat(folder_dir, nameFolds{i}, '/myeloma_cca*.mat'));
    cca_filename = cca_file.name;
    clusters = load(strcat(folder_dir, nameFolds{i}, '/', cca_filename));

    cluster_names = fieldnames(clusters);
    for j=1:numel(cluster_names)
        cluster = clusters.(cluster_names{j});
        [enrich_data, enrich_genes] = geneset_enrichment(genes.genes, cluster, 'GO-BP', strcat(folder_dir, nameFolds{i}, '/cca_', char(cluster_names{j})), 20);
    end
end