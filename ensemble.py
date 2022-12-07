import os
import pandas as pd
import numpy as np
from ensembles.ensembles import Ensemble
import argparse
from pytz import timezone
from datetime import datetime


def main(args):
    filepath = args.FILE_PATH
    savepath = args.RESULT_PATH
    
    os.makedirs(filepath, exist_ok=True)
    if os.listdir(filepath) == []:
        raise ValueError(f"Put inference.csv files in folder path {filepath}")
    os.makedirs(savepath, exist_ok=True)
    
    en = Ensemble(filepath=filepath)
    
    if args.ENSEMBLE_STRATEGY == 'WEIGHTED':
        if args.ENSEMBLE_WEIGHT: 
            strategy_title = 'sw-'+'-'.join(map(str,*args.ENSEMBLE_WEIGHT)) #simple weighted
            result = en.simple_weighted(*args.ENSEMBLE_WEIGHT)
        else:
            strategy_title = 'aw' #average weighted
            result = en.average_weighted()
    elif args.ENSEMBLE_STRATEGY == 'MIXED':
        strategy_title = args.ENSEMBLE_STRATEGY.lower() #mixed
        result = en.mixed()
    elif args.ENSEMBLE_STRATEGY =='HARDSOFT':
        strategy_title = 'hs' #hardsoft
        result = en.hardsoft()
    else:
        pass
    en.output_frame['prediction'] = result
    output = en.output_frame.copy()
    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y-%m-%d_%H:%M")

    output.to_csv(f'{savepath}{now}-{strategy_title}.csv',index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    
    arg("--FILE_PATH", type=str,required=False,
        default="./ensembles_inference/",
        help='required: 앙상블 하고 싶은 inference 파일들이 있는 폴더의 경로를 입력해주세요.')
    arg('--ENSEMBLE_STRATEGY', type=str, default='WEIGHTED',
        choices=['WEIGHTED','MIXED','HARDSOFT'],
        help='optional: [MIXED, WEIGHTED, HARDSOFT] 중 앙상블 전략을 선택해 주세요. (default="WEIGHTED")')
    arg('--ENSEMBLE_WEIGHT', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: Weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.')
    arg('--RESULT_PATH',type=str, default='./ensembles_submit/',
        help='optional: 앙상블 결과를 저장할 폴더 경로를 전달합니다. (default:"./ensembles_submit/")')
    args = parser.parse_args()
    
    main(args)

