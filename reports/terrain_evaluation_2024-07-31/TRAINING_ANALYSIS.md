# Go2訓練環境分析レポート

## 訓練スクリプト詳細 (examples/locomotion/go2_train.py)

### 1. 訓練環境の制約

#### 地形設定
```python
# go2_env.py line 52
self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
```
**結果**: 完全な平地でのみ訓練、地形多様性ゼロ

#### 観測空間 (45次元)
```python
# go2_env.py lines 170-180
obs = torch.cat([
    self.base_ang_vel * self.obs_scales["ang_vel"],  # 3: 角速度
    self.projected_gravity,                          # 3: 重力方向  
    self.commands * self.commands_scale,             # 3: 速度コマンド
    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"], # 12: 関節偏差
    self.dof_vel * self.obs_scales["dof_vel"],       # 12: 関節速度
    self.actions,                                    # 12: 前回アクション
])
```
**地形関連情報**: なし（高さマップ、接触情報、傾斜情報すべて欠如）

### 2. 制御パラメータ

#### PD制御設定
```python
# go2_train.py lines 102-103
"kp": 20.0,
"kd": 0.5,
```

#### アクション処理
```python  
# go2_env.py lines 121-124
self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
exec_actions = self.last_actions if self.simulate_action_latency else self.actions
target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
# action_scale = 0.25 (line 112)
```

#### 初期姿勢
```python
# go2_train.py line 108
"base_init_pos": [0.0, 0.0, 0.42],
```

### 3. 報酬関数設計

#### 報酬スケール
```python
# go2_train.py lines 129-136
reward_scales = {
    "tracking_lin_vel": 1.0,     # 速度追従（平地前提）
    "tracking_ang_vel": 0.2,     # 旋回追従
    "lin_vel_z": -1.0,           # Z軸速度ペナルティ
    "base_height": -50.0,        # 高さ維持（平地特化）
    "action_rate": -0.005,       # アクション変化ペナルティ
    "similar_to_default": -0.1,  # デフォルト姿勢維持
}
```

**地形適応報酬**: なし（足先クリアランス、接触力制御、地形認識なし）

#### 速度コマンド
```python
# go2_train.py lines 140-142  
"lin_vel_x_range": [0.5, 0.5],  # 固定前進速度
"lin_vel_y_range": [0, 0],       # 横移動なし
"ang_vel_range": [0, 0],         # 旋回なし
```

### 4. PPOアルゴリズム設定

#### ネットワーク構造
```python
# go2_train.py lines 42-46
"policy": {
    "activation": "elu",
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "init_noise_std": 1.0,
},
```

#### 学習パラメータ
```python
# go2_train.py lines 24-38
"algorithm": {
    "class_name": "PPO",
    "clip_param": 0.2,
    "desired_kl": 0.01,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "lam": 0.95,
    "learning_rate": 0.001,
    "max_grad_norm": 1.0,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
},
```

### 5. 訓練結果の制約

#### 学習した能力
- ✅ 平地での安定歩行
- ✅ 0.5m/s前進速度維持
- ✅ バランス制御（平地のみ）
- ✅ 関節協調動作

#### 学習できなかった能力  
- ❌ 地形認識・分類
- ❌ 高さ変化への適応
- ❌ 接触力に基づく制御調整
- ❌ 動的な速度調整
- ❌ 障害物回避・克服

### 6. 汎化失敗の根本原因

#### 環境制約
1. **単一地形**: plane.urdfのみで地形多様性なし
2. **観測不足**: 地形情報が観測空間に含まれない
3. **報酬偏重**: base_height(-50.0)が地形適応を阻害

#### アーキテクチャ制約
1. **固定コマンド**: 地形に応じた速度調整不可
2. **平地特化報酬**: 地形適応インセンティブなし
3. **観測次元不足**: 45次元では地形情報を表現できない

### 7. 改善のための具体的変更点

#### 即座の修正 (評価ツール)
```python
# 1. 正しい観測実装
obs = torch.cat([
    base_ang_vel * obs_scales["ang_vel"],
    projected_gravity,  # 実際の重力方向を計算
    commands * commands_scale,  # [0.5, 0.0, 0.0]
    (dof_pos - default_dof_pos) * obs_scales["dof_pos"], 
    dof_vel * obs_scales["dof_vel"],
    actions,  # 前回アクション履歴
])

# 2. アクションスケール適用
target_dof_pos = actions * 0.25 + default_dof_pos

# 3. 初期姿勢調整
robot_pos = [0.0, 0.0, 0.42]  # 訓練時と同じ
```

#### 中期的改修 (訓練フレームワーク)
```python
# 1. 地形ランダム化
terrain_types = ["flat", "steps_5cm", "steps_10cm", "slopes_low"]
terrain = scene.add_entity(gs.morphs.Terrain(...))

# 2. 観測拡張 (45 -> 60次元)
obs = torch.cat([
    # 既存観測 (45次元)
    base_ang_vel, projected_gravity, commands, dof_pos, dof_vel, actions,
    # 地形関連観測 (15次元追加)
    height_measurements,  # 8: 足先周辺高さ
    contact_states,       # 4: 各脚接触状態
    terrain_normal,       # 3: 地面法線ベクトル
])

# 3. 地形適応報酬
reward_scales.update({
    "base_height": -10.0,        # 平地特化を緩和
    "foot_clearance": 0.1,       # 足上げ時クリアランス
    "contact_timing": 0.2,       # 適切な接触タイミング
    "terrain_adaptation": 0.5,   # 地形に応じた動作
})
```

---

**結論**: Go2の訓練環境は完全に平地特化されており、地形汎化には根本的な改修が必要。評価ツールでの観測実装修正は短期的対策として有効だが、真の地形適応にはgo2_train.py自体の大幅改修が不可欠。