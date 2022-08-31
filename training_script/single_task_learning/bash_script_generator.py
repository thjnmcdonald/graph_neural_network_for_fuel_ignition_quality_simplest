def bash_script_generator():
    with open('training_script/single_task_learning/script.bash', 'a') as bash_script:
        for gnn in ['GCN', 'GraphConv']:
            for i in range(30):
                for layers in range(1,5):
                    for nodes in [16, 32, 64]:
                        bash_script.write(f'python training_singletask_model_5_final_better_reporting.py --gnn {gnn} --num_layers {layers} --num_nodes {nodes} \n')


def bash_script_generator_2(name, reps, num_layers, node_list):
    with open(f'{name}.bash', 'w+') as bash_script:
        for gnn in ['GCN', 'GraphConv']:
            for layers in range(1,num_layers):
                for nodes in node_list:
                    for i in range(reps):
                        bash_script.write(f'python training_singletask_model_5_final_better_reporting.py --gnn {gnn} --num_layers {layers} --num_nodes {nodes} \n')


bash_script_generator_2('15more', 15, 4, [16, 32, 64])                
