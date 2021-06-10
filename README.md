# Stock_Trader
실행 방법

환경
Python 3.6
tensorflow 1.15
Plaid_ML

Ranbow DQN을 사용하기 위해서는 Plaid_ML을 설치 하셔야합니다.  
Train  
각 원하는 알고리즘 경로로 들어 간뒤  
명령어 실행 창에서  
python main.py --stock_code [종목코드] --rl_method [알고리즘] --net [ ] --num_steps [ ] --lr [ ] --discount_factor [ ] --learning --start_epslion [ ]--start_date [시작 날짜 ] --end_date [끝없는 날짜]  --output_name [ ]     
  
--net 같은 경우는 ddpg와 TD3는 지우기Rainbow만 활용  
[]을 대신에 예시 처럼 입력  
DDPG  
ex) python main.py --stock_code XOM --rl_method ddpg --num_steps 5 --lr 0.0001 --discount_factor 0.99 --learning  —num_epoches 1000 —start_epsilon 1--start_date 20150101 -->end_date 20191231 --output_name XOM_ddpg  

Rainbow  
ex) python main.py —stock_code XEC —rl_method rainbow —net dndlstm --lr 0.0001 --discount_factor 0.99 --learning  —num_steps 5  —num_epoches 1000 —start_epsilon 1 —start_date 20150101 —end_date 20191231 —output_name Rainbow_XEC_T2 —backend plaidml  

Test  
각 원하는 알고리즘 경로로 들어 간뒤  
명령어 실행 창에서  
python main.py --stock_code [종목코드] --rl_method [알고리즘] --net [  ] --num_steps [ ] --reuse_models --start_date [시작 날짜 ] --end_date [끝없는 날짜] --policy_network_path >[ ]  --value_network_path [ ] --output_name [ ]  
DDPG TD3 같은 경우 Policy이기 때문에 --value_network_path는 지우고 --policy_network_path 만 사용  
RainBow는 반대로 --policy_network_path만 사용 --value_network_path만 작성  
  
[]을 대신에 예시 처럼 입력  
DDPG  
ex) python main.py --stock_code XOM --rl_method ddpg --num_steps 5 --reuse_models --start_date 20200501 --end_date 20210501 --policy_network_path XOM_ddpg.h5 --output_name Test_XOM_ddpg  

Rainbow  
ex) python main.py —stock_code XEC —rl_method rainbow —net dndlstm —num_steps 5 —output_name test_XEC_T2 —num_epoches 1 —start_epsilon 0 —start_date 20200501 —end_date 20210501 —reuse_models —value_network_name rainbow_dndlstm_value_c_XEC —backend plaidml  
