"""
メモリ効率化機能のユニットテスト
"""

import unittest

import torch

from style_bert_vits2.models.memory_efficient import (
    allocate_bucketed_tensor,
    clear_memory_pools,
    copy_to_bucketed_tensor,
    get_bucket_size,
    get_memory_pool_stats,
)


class TestMemoryEfficient(unittest.TestCase):
    """メモリ効率化機能のテストケース"""

    def setUp(self):
        """各テストの前に実行"""
        clear_memory_pools()

    def tearDown(self):
        """各テストの後に実行"""
        clear_memory_pools()

    def test_get_bucket_size(self):
        """バケツサイズ選択のテスト"""
        # 基本的なケース（max_overhead_ratio=1.5）
        self.assertEqual(
            get_bucket_size(10), 10
        )  # 10 * 1.5 = 15 < 16は大きすぎるので10を返す
        self.assertEqual(get_bucket_size(16), 16)
        self.assertEqual(get_bucket_size(17), 17)  # 17 * 1.5 = 25.5 < 32なので17を返す
        self.assertEqual(get_bucket_size(50), 64)  # 50 * 1.5 = 75 > 64
        self.assertEqual(get_bucket_size(100), 128)  # 100 * 1.5 = 150 > 128

        # オーバーヘッド比率の制限
        # 10 * 1.6 = 16 なので、16は許容範囲内
        self.assertEqual(get_bucket_size(10, max_overhead_ratio=1.6), 16)
        # 10 * 1.5 = 15 なので、16は大きすぎる → 10を返すべき
        self.assertEqual(get_bucket_size(10, max_overhead_ratio=1.5), 10)

        # バケツサイズを超える場合
        self.assertEqual(
            get_bucket_size(1000), 1024
        )  # 1000 * 1.5 = 1500 < 1536なので1024を返す
        self.assertEqual(get_bucket_size(3000), 3000)  # 2048を超えるので実長を返す

    def test_allocate_bucketed_tensor(self):
        """バケツ化されたテンソル割り当てのテスト"""
        device = torch.device("cpu")

        # 基本的な割り当て
        tensor, bucket_size = allocate_bucketed_tensor(
            shape=(1, 10),
            actual_length=10,
            length_dim=1,
            dtype=torch.float32,
            device=device,
            use_pool=False,
        )

        self.assertEqual(
            tensor.shape, (1, 10)
        )  # 10は実長のまま（1.5倍以内のバケツがない）
        self.assertEqual(bucket_size, 10)
        # torch.emptyなので初期化されていない（内容は不定）

    def test_copy_to_bucketed_tensor(self):
        """バケツ化テンソルへのコピーのテスト"""
        device = torch.device("cpu")

        # ソーステンソル作成
        source = torch.randn(1, 10, 5)

        # length_dim=1でコピー
        bucketed, actual_length = copy_to_bucketed_tensor(
            source, length_dim=1, use_pool=False
        )

        self.assertEqual(bucketed.shape, (1, 10, 5))  # 10は実長のまま
        self.assertEqual(actual_length, 10)

        # データが正しくコピーされているか確認
        torch.testing.assert_close(bucketed[:, :10, :], source)

        # パディング部分なし（実長のまま）

    def test_memory_pool(self):
        """メモリプール機能のテスト"""
        device = torch.device("cpu")
        model_name = "test_model"

        # 初回割り当て（16にバケツ化されるサイズ）
        tensor1, _ = allocate_bucketed_tensor(
            shape=(1, 16),
            actual_length=16,
            length_dim=1,
            dtype=torch.float32,
            device=device,
            use_pool=True,
        )

        # プールに追加されているか確認（dtype別管理になったので確認方法が変わる）
        stats = get_memory_pool_stats()
        # 少なくとも1つのプールが存在することを確認
        self.assertTrue(len(stats) > 1)  # _totalキーも含まれるため

        # 同じサイズの再割り当て（プールから取得）
        tensor1[0, 0] = 1.0  # マーカー値を設定

        tensor2, _ = allocate_bucketed_tensor(
            shape=(1, 16),
            actual_length=16,
            length_dim=1,
            dtype=torch.float32,
            device=device,
            use_pool=True,
        )

        # プールから取得されたが、内容はクリアされていない（torch.emptyのまま）
        # ただし同じ形状のテンソルが返される
        self.assertEqual(tensor2.shape, tensor1.shape)

        # プールクリア
        clear_memory_pools(model_name)
        stats = get_memory_pool_stats()
        # プールがクリアされたことを確認
        self.assertEqual(len(stats), 1)  # _totalキーのみ残る

    def test_different_length_dims(self):
        """異なる長さ次元でのコピーのテスト"""
        device = torch.device("cpu")

        # length_dim=0
        source = torch.randn(10, 5)
        bucketed, _ = copy_to_bucketed_tensor(source, length_dim=0, use_pool=False)
        self.assertEqual(bucketed.shape, (10, 5))  # 実長のまま
        torch.testing.assert_close(bucketed[:10], source)

        # length_dim=2
        source = torch.randn(3, 4, 10)
        bucketed, _ = copy_to_bucketed_tensor(source, length_dim=2, use_pool=False)
        self.assertEqual(bucketed.shape, (3, 4, 10))  # 実長のまま
        torch.testing.assert_close(bucketed[:, :, :10], source)

        # length_dim=-1 (最後の次元)
        source = torch.randn(3, 4, 10)
        bucketed, _ = copy_to_bucketed_tensor(source, length_dim=-1, use_pool=False)
        self.assertEqual(bucketed.shape, (3, 4, 10))  # 実長のまま
        torch.testing.assert_close(bucketed[..., :10], source)


class TestBucketIntegration(unittest.TestCase):
    """推論との統合テスト"""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_memory_pool(self):
        """CUDA環境でのメモリプールテスト"""
        device = torch.device("cuda")

        # テンソル割り当て
        tensor, _ = allocate_bucketed_tensor(
            shape=(1, 100, 512),
            actual_length=100,
            length_dim=1,
            dtype=torch.float32,
            device=device,
            use_pool=True,
        )

        self.assertEqual(tensor.device.type, "cuda")
        self.assertEqual(tensor.shape, (1, 128, 512))  # 100 → 128にバケツ化

        # メモリプールクリア
        clear_memory_pools()


if __name__ == "__main__":
    unittest.main()
