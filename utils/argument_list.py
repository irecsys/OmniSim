# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@iit.edu

"""
mopo.utils.argument_list
################################################
"""


dataset_arguments = [
    'data_path', 'dataset', 'file_items', 'file_users', 'file_interactions', 
    'col_userid', 'col_title', 'col_category', 'col_details', 'col_itemid', 'col_rating',
    'default_attributes', 'item_attributes'
]

simulation_arguments = [
    'role_user', 'role_bot', 'users_as_guest', 'mode_refinement',
    'threshold_similarity', 'max_similar_items', 'max_rec_attempts', 'rec_top_n',
    'guestuser_randomitem_records', 'chats_per_entry',
    'generation_strategy', 'input_items_file', 'input_pairs_file',
    'chit_chat_ratio', 'rejection_explanation_ratio',
    'rating_threshold', 'short_term_days',
    'weight_es_score', 'weight_user_taste_short',
]

outputs_arguments = [
    'name_bot', 'name_user', 'annotation', 'outputs_folder', 'log_path_outputs',
]

general_arguments = [
    'seed', 'library_name', 'library_version'
]

openai_arguments = [
    'openai_provider', 'chat_model', 'embeddings_model',
    'chat_endpoint', 'chat_api_version', 'embeddings_endpoint', 'embeddings_api_version'
]

