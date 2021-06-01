import re
from omegaconf import OmegaConf, DictConfig


def register():
    OmegaConf.register_resolver("concat", lambda x, y: x+y)
    OmegaConf.register_resolver("add", lambda x, y: int(x) + int(y))
    OmegaConf.register_resolver("sub", lambda x, y: int(x) - int(y))
    OmegaConf.register_resolver("mul", lambda x, y: int(x) * int(y))
    OmegaConf.register_resolver("div", lambda x, y: int(x) // int(y))


def try_evaluate(cfg: DictConfig, head: DictConfig):
    """
    config.yaml에 (eval) 표시가 있으면, eval 함수로 계산
    """
    num_evals_left = 0
    for key in cfg.keys():
        if type(cfg[key]) == str and '(eval)' in cfg[key]:
            try:
                print('(before) key: {} / val: {} \n[{}]'.format(key, cfg[key], cfg))

                pattern = "(<replace>)(.*)(</replace>)"
                # 안에 <replace>이 있으면 먼저 연산 후,
                replace_expr = re.search(pattern, cfg[key])
                if replace_expr is not None:
                    evaluated = eval("head." + re.sub(pattern, "\\2", replace_expr.group()))
                    # 연산했는데 연산이 안되고 여전히 string이면 일단 스킵 ('a * 2' = 'aa'가 되는 상황 방지)
                    if isinstance(evaluated, str):
                        num_evals_left += 1
                        continue

                    # <replace> 표현을 값으로 치환
                    cfg[key] = re.sub(pattern, str(evaluated), cfg[key])

                # 치환된 표현 계산
                cfg[key] = eval(cfg[key].split('(eval)')[-1])
                print('(after) key: {} / val: {}'.format(key, cfg[key]))
            except:
                print('(retry next)')
                num_evals_left += 1
                continue

        elif type(cfg[key]) == DictConfig:
            num_evals_left += try_evaluate(cfg[key], head)

    return num_evals_left


def evaluate_cfg(cfg: DictConfig):
    i = 0
    while True:
        i += 1
        num_evals_left = try_evaluate(cfg, cfg)
        if num_evals_left == 0 or i > 10:
            break

    print(i)
