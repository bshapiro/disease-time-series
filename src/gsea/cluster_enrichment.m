genes = load('../../data/myeloma_genes.mat');
clusters = load('../../data/myeloma_pca_clusters.mat');
cluster_names = fieldnames(clusters);
for i=1:numel(cluster_names)
    cluster = clusters.(cluster_names{i});
    [enrich_data, enrich_genes] = geneset_enrichment(genes.genes, cluster, 'GO-BP', strcat('../../data/gsea_output/svd_', char(cluster_names{i})), 20);
end

clusters = load('../../data/myeloma_cca_clusters.mat');
cluster_names = fieldnames(clusters);
for i=1:numel(cluster_names)
    cluster = clusters.(cluster_names{i});
    [enrich_data, enrich_genes] = geneset_enrichment(genes.genes, cluster, 'GO-BP', strcat('../../data/gsea_output/cca_', char(cluster_names{i})), 20);
end
