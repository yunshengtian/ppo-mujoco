import re
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_BENCHMARKS = []

remove_version_re = re.compile(r'-v\d+$')


def register_benchmark(benchmark):
    for b in _BENCHMARKS:
        if b['name'] == benchmark['name']:
            raise ValueError('Benchmark with name %s already registered!' % b['name'])

    # automatically add a description if it is not present
    if 'tasks' in benchmark:
        for t in benchmark['tasks']:
            if 'desc' not in t:
                t['desc'] = remove_version_re.sub('', t.get('env_id', t.get('id')))
    _BENCHMARKS.append(benchmark)


def list_benchmarks():
    return [b['name'] for b in _BENCHMARKS]


def get_benchmark(benchmark_name):
    for b in _BENCHMARKS:
        if b['name'] == benchmark_name:
            return b
    raise ValueError('%s not found! Known benchmarks: %s' % (benchmark_name, list_benchmarks()))


def get_task(benchmark, env_id):
    """Get a task by env_id. Return None if the benchmark doesn't have the env"""
    return next(filter(lambda task: task['env_id'] == env_id, benchmark['tasks']), None)


def find_task_for_env_id_in_any_benchmark(env_id):
    for bm in _BENCHMARKS:
        for task in bm["tasks"]:
            if task["env_id"] == env_id:
                return bm, task
    return None, None


# MuJoCo

_mujocosmall = [
    'InvertedDoublePendulum-v2', 'InvertedPendulum-v2',
    'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2',
    'Reacher-v2', 'Swimmer-v2']
register_benchmark({
    'name': 'Mujoco1M',
    'description': 'Some small 2D MuJoCo tasks, run for 1M timesteps',
    'tasks': [{'env_id': _envid, 'trials': 6, 'num_timesteps': int(1e6)} for _envid in _mujocosmall]
})

register_benchmark({
    'name': 'MujocoWalkers',
    'description': 'MuJoCo forward walkers, run for 8M, humanoid 100M',
    'tasks': [
        {'env_id': "Hopper-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "Walker2d-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "Humanoid-v1", 'trials': 4, 'num_timesteps': 100 * 1000000},
    ]
})

# Bullet
_bulletsmall = [
    'InvertedDoublePendulum', 'InvertedPendulum', 'HalfCheetah', 'Reacher', 'Walker2D', 'Hopper', 'Ant'
]
_bulletsmall = [e + 'BulletEnv-v0' for e in _bulletsmall]

register_benchmark({
    'name': 'Bullet1M',
    'description': '6 mujoco-like tasks from bullet, 1M steps',
    'tasks': [{'env_id': e, 'trials': 6, 'num_timesteps': int(1e6)} for e in _bulletsmall]
})


# Roboschool

register_benchmark({
    'name': 'Roboschool8M',
    'description': 'Small 2D tasks, up to 30 minutes to complete on 8 cores',
    'tasks': [
        {'env_id': "RoboschoolReacher-v1", 'trials': 4, 'num_timesteps': 2 * 1000000},
        {'env_id': "RoboschoolAnt-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolHalfCheetah-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolHopper-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
        {'env_id': "RoboschoolWalker2d-v1", 'trials': 4, 'num_timesteps': 8 * 1000000},
    ]
})
register_benchmark({
    'name': 'RoboschoolHarder',
    'description': 'Test your might!!! Up to 12 hours on 32 cores',
    'tasks': [
        {'env_id': "RoboschoolHumanoid-v1", 'trials': 4, 'num_timesteps': 100 * 1000000},
        {'env_id': "RoboschoolHumanoidFlagrun-v1", 'trials': 4, 'num_timesteps': 200 * 1000000},
        {'env_id': "RoboschoolHumanoidFlagrunHarder-v1", 'trials': 4, 'num_timesteps': 400 * 1000000},
    ]
})


# HER DDPG

_fetch_tasks = ['FetchReach-v1', 'FetchPush-v1', 'FetchSlide-v1']
register_benchmark({
    'name': 'Fetch1M',
    'description': 'Fetch* benchmarks for 1M timesteps',
    'tasks': [{'trials': 6, 'env_id': env_id, 'num_timesteps': int(1e6)} for env_id in _fetch_tasks]
})

