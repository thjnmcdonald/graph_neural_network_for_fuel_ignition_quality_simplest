with open('training_script/single_task_learning/script.bash', 'a') as bash_script:
    for gnn in ['GCN', 'GraphConv']:
        for i in range(30):
            for layers in range(1,5):
                for nodes in [16, 32, 64]:
                    bash_script.write(f'python training_singletask_model_5_final_better_reporting.py --gnn {gnn} --num_layers {layers} --num_nodes {nodes} \n')