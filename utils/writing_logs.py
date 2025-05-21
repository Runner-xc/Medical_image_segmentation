def writing_logs(writer, train_metrics, val_metrics, epoch, class_names):
    """-------------------------Loss--------------------------------------------------------""" 
    for index, name in enumerate(class_names):
        writer.add_scalars(f'Loss/{name}',
                        {'train':train_metrics['Loss'][index],
                        'Val':val_metrics['Loss'][index]},
                        epoch)              
                    
    """-------------------------Dice--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'Dice/{name}',
                        {'train':train_metrics['Dice'][index],
                            'Val':val_metrics['Dice'][index]},
                            epoch)
    
    """-------------------------Precision--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'Precision/{name}',
                        {'train':train_metrics['Precision'][index],
                            'Val':val_metrics['Precision'][index]},
                            epoch)
    
    """-------------------------Recall--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'Recall/{name}',
                        {'train':train_metrics['Recall'][index],
                            'Val':val_metrics['Recall'][index]},
                            epoch)
    
    """-------------------------F1_scores--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'F1_scores/{name}',
                        {'train':train_metrics['F1_scores'][index],
                            'Val':val_metrics['F1_scores'][index]},
                            epoch)
    
    """-------------------------mIoU--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'mIoU/{name}',
                        {'train':train_metrics['mIoU'][index],
                            'Val':val_metrics['mIoU'][index]},
                            epoch)

    """-------------------------Accuracy--------------------------------------------------------"""
    for index, name in enumerate(class_names):
        writer.add_scalars(f'Accuracy/{name}',
                        {'train':train_metrics['Accuracy'][index],
                            'Val':val_metrics['Accuracy'][index]},
                            epoch)
 
            