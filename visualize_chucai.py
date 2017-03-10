from __future__ import print_function
from pprint import pprint
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pandas

datapath = 'data/'

# posefiles = [i for i in os.listdir(datapath) if i.endswith('csv')]
# print('found the following pose files:')
# pprint(posefiles)
# posefile = posefiles[0]
# x = numpy.loadtxt(datapath+posefile)
# plt.plot(x[:,0],x[:,1])

# Abbreviations:
# 	ss......subsection (of map)
# 	acc.....accessor (separator)
# 	lb......lane boundary

subsections = [i for i in os.listdir(datapath) if os.path.isdir(datapath+i)]

ss_acc_map = {int(ss):set() for ss in subsections}
ss_timestamp_start = {}
ss_timestamp_end = {}

# build the {ss -> acc} map
for ss in subsections:
	es_list = [i for i in os.listdir(datapath+ss) if i[:2]=='ES' and i.endswith('csv')]
	for es_file in es_list:
		acc_set = set([ int(es[4:]) for es in es_file[:-4].split('-') if es[:2]=='SB' ])
		ss_acc_map[int(ss)] |= acc_set
		es_dataframe = pandas.read_csv(	datapath+ss+'/'+es_file, sep=',\s+', engine='python' )
		ss_timestamp_start[int(ss)] = es_dataframe['StartTime'][0]
		ss_timestamp_end[int(ss)] = es_dataframe['EndTime'][0]

# build the inverse map {acc -> ss}
acc_ss_map = {acc:set() for acc in set.union(*ss_acc_map.values())}
for ss, acc_set in ss_acc_map.iteritems():
	for acc in acc_set:
		acc_ss_map[acc].add(ss)

# build the adjacency graph of subsections {ss <-> ss}
ss_graph = {int(ss):set() for ss in subsections}
for acc in acc_ss_map:
	ss_list = list(acc_ss_map[acc])
	if len(ss_list)>1:
		assert len(ss_list)<3, "accessor cannot connect 3 or more subsections!"
		ss_graph[ss_list[0]] |= set([ss_list[1]])
		ss_graph[ss_list[1]] |= set([ss_list[0]])

# at least two nodes must have degree of 1
end_nodes = [node for node in ss_graph if len(ss_graph[node])==1]
current_node = end_nodes[0]
ss_list = [current_node]
while current_node != end_nodes[1]:
	assert len(ss_graph[current_node])==1, "subsection can have either 1 or 2 neighbors!"
	next_node = ss_graph[current_node].pop()
	ss_graph[next_node].remove(current_node)
	current_node = next_node
	ss_list.append(current_node)

# read lane boundaries for a segment of the drive
ss_lb_dict = {int(ss):[] for ss in subsections}
subsections_sorted = ['%d'%ss for ss in ss_list]
for ss in subsections_sorted:
	lb_list = [i for i in os.listdir(datapath+ss) if (i[:2]=='LB' or i[:4]=='null') and i.endswith('csv')]
	for lb_file in lb_list:
		lb_dataframe = pandas.read_csv(	datapath+ss+'/'+lb_file,
			skiprows=[0],
			skipfooter=1,
			header=None,
			sep=' ',
			engine='python')
		ss_lb_dict[int(ss)].append((lb_file,numpy.array(lb_dataframe.iloc[:,:2])))

# display extracted lane boundaries
for ss in subsections_sorted:
	for lb in ss_lb_dict[int(ss)]:
		plt.plot(lb[1][:,0],lb[1][:,1],linewidth=.01)
plt.savefig('trajectory.pdf')


lines = []
for ss in ss_list:
	if not lines:
		for lb in ss_lb_dict[ss]:
			lines.append(lb[1][:2])
	for lb in ss_lb_dict[ss]:
		# lb_filename = lb[0]
		# t0, t1 = [float(i) for i in lb_filename[:-4].split('-')[-2:]]
		lb_data = lb[1]
		for line in lines:
			if line[-1]