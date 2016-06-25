import numpy as np
from biomart import BiomartServer
import pickle

class BioMartSearch:

	def __init__ (self, data, server, dataset, filters, attributes):
		self.server = BiomartServer(server)
		self.dataset = self.server.datasets[dataset]
		self.data = data
		self.filters = filters
		self.attributes = attributes
		self.dictionary = self.makedictionary()


	def makedictionary(self):
		print 'Prepping Dictionary...'
		dictionary = {entry: '' for entry in self.data}
		return dictionary


	def searchbiomart(self, n=-1):
		"""
		Searches BioMart with an itesearrable collection of accesion_codes
		right now it only chekcs for embl and refseq_mrna accession codes
		but just use the ensembl biomart tool to find out what other filters
		we could use
		Returns a dictionary mapping accession codes to gene names
		"""
		print 'Searching Biomart...'
		keys = self.dictionary.keys()
		if n == -1:
			n = len(keys)

		for i in range(0, n):
			self.dictionary[keys[i]] = self.singlesearch(keys[i])

		pickle.dump(self.dictionary, open('gene_acc_to_gene_names.p', 'wb'))

	def bulksearch(self, filt):
		dkeys = self.dictionary.keys()
		keys = [dkeys[i:i+200] for i in range(0, len(dkeys), 200)]
		print len(keys), 'batches'
		batch = 1
		for keyset in keys:
			print batch
			batch += 1
			response = self.dataset.search({
			'filters' : { filt: keyset},
			'attributes': self.attributes
			})
			self.dictionary.update({key: val for key, val in zip(keyset, response.text.split('\n'))})	

	def singlesearch(self, accession_code):
		"""
		makes a single search
		"""
		print accession_code
		response = np.empty(len(self.filters), dtype=object)
		index = 0
		# search each of the potential features
		for filter in self.filters:
			response[index] = self.dataset.search({
			'filters' : { filter : [accession_code]},
			'attributes': self.attributes
			})
			index += 1

		# pick one of the responses that's non-empty
		best_out = ''
		for i in range(0, response.size):
			if best_out < response[i].text:
				best_out = response[i].text

		# return the output stripped of whitespace
		return best_out.strip()