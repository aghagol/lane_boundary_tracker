# from __future__ import print_function
# from pprint import pprint
import sys, os
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pandas

datapath = 'data/HT073_1463746090/'
# datapath = 'data/HT080_1465383546/'
# datapath = 'data/HT085_1465209329/'

pose_file_map = {
	'data/HT085_1465209329/': 'data/80379367-pose.csv',
	'data/HT080_1465383546/': 'data/80379517-pose.csv',
	'data/HT073_1463746090/': 'data/79985746-pose.csv'	}

# read pose file and plot the pose
posefile = pose_file_map[datapath]
pose_llat = numpy.loadtxt(posefile)
pose_n = pose_llat.shape[0]
pose_x = pose_llat[:,0]
pose_y = pose_llat[:,1]
# plt.plot(pose_x[::100], pose_y[::100], '^')
plt.plot(pose_x, pose_y, linewidth=.01)

# # show animation of pose
# ww = 1000 # view window width for animation
# for t in range(2000,pose_n,100):
# 	plt.clf()
# 	plt.plot(pose_x[max(0,t-ww):min(t+ww,pose_n)], pose_y[max(0,t-ww):min(t+ww,pose_n)], linewidth=.1)
# 	plt.plot(pose_x[t],pose_y[t],'ro')
# 	plt.pause(0.001)

# Abbreviations:
# 	ss...........subsection (of map)
# 	acc..........accessor (separator)
# 	lb...........lane boundary

subsections = [i for i in os.listdir(datapath) if os.path.isdir(datapath+i)]

ss_acc_map = {int(ss):set() for ss in subsections}
ss_timestamp_start = {}
ss_timestamp_end = {}

# build map: ss -> acc
for ss in subsections:
	es_list = [i for i in os.listdir(datapath+ss) if i[:2]=='ES' and i.endswith('csv')]
	for es_file in es_list:
		acc_set = set([ int(es[4:]) for es in es_file[:-4].split('-') if es[:2]=='SB' ])
		ss_acc_map[int(ss)] |= acc_set # | is union operator
		es_dataframe = pandas.read_csv(	datapath+ss+'/'+es_file, sep=',\s+', engine='python' )
		ss_timestamp_start[int(ss)] = es_dataframe['StartTime'][0]
		ss_timestamp_end[int(ss)] = es_dataframe['EndTime'][0]

# build map: acc -> ss
acc_ss_map = {acc:set() for acc in set.union(*ss_acc_map.values())}
for ss, acc_set in ss_acc_map.iteritems():
	for acc in acc_set:
		acc_ss_map[acc].add(ss)

# build adjacency graph of subsections {ss <-> ss}
ss_graph = {int(ss):set() for ss in subsections}
for acc in acc_ss_map:
	ss_list = list(acc_ss_map[acc])
	if len(ss_list)>1:
		assert len(ss_list)<3, "accessor cannot connect 3 or more subsections!"
		ss_graph[ss_list[0]] |= set([ss_list[1]])
		ss_graph[ss_list[1]] |= set([ss_list[0]])

# compute a Hamiltonian path of the ss graph
end_nodes = [node for node in ss_graph if len(ss_graph[node])==1] # find the leaf nodes
if len(end_nodes) > 2 :
	print("Looks like we have intersections!")
	print("Trimming the intersections by keeing the longest path...")
	end_nodes_sorted = sorted(end_nodes, key=ss_timestamp_start.get)
	end_nodes = [end_nodes_sorted[0], end_nodes_sorted[-1]]
	# for v in end_nodes_sorted[1:-1]:
	# 	ss_graph[v] = set()
	print("Done")

current_node = end_nodes[0] # pick one leaf node to start from
ss_list = [current_node] # sorted list of nodes. initialize with leaf.
while current_node != end_nodes[1]:
	assert len(ss_graph[current_node])==1, "No intersections allowed!"
	next_node = ss_graph[current_node].pop()
	ss_graph[next_node].remove(current_node)
	current_node = next_node
	ss_list.append(current_node)

# read lane boundaries for a segment of the drive
print("Reading Lane Boundaries")
ss_lb_dict = {ss:[] for ss in ss_list}
subsections_sorted = ['%d'%ss for ss in ss_list] # convert to string
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
print("Done")

# display extracted lane boundaries
print("Plotting...")
for ss in subsections_sorted:
	for lb in ss_lb_dict[int(ss)]:
		plt.plot(lb[1][:,0],lb[1][:,1],linewidth=.05)
plt.savefig('trajectory.svg')
print("Done")


# lines = []
# for ss in ss_list:
# 	if not lines:
# 		for lb in ss_lb_dict[ss]:
# 			lines.append(lb[1][:2])
# 	for lb in ss_lb_dict[ss]:
# 		# lb_filename = lb[0]
# 		# t0, t1 = [float(i) for i in lb_filename[:-4].split('-')[-2:]]
# 		lb_data = lb[1]
# 		for line in lines:
# 			if line[-1]