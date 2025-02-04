import torch
import numpy as np

class DDPMSampler:
    def __init__(self, 
                 generator: torch.Generator,
                 num_training_steps: int=1000,
                 beta_start: float=0.00085, 
                 beta_end: float=0.0120):
        """
        Xây dựng lịch trình nhiễu giúp kiểm sóat mức độ nhiễu hóa ảnh theo thời gian.
        Args:
            betas: giá trị tỉ lệ nhiễu theo từng bước thời gian
            alphas: phần giữ lại thông tin gốc sau mỗi bước
            alphas_cumprod: theo dõi mức độ nhiễu tích lũy theo thời gian 

        Các giá trị này giúp kiểm soát lượng nhiễu thêm vào ảnh tại mỗi bước, đảm bảo nhiễu hóa dần dần thay vì ngẫu nhiên.
        """
        # [num_traning_steps, ]
        self.beta = torch.linspace(start=beta_start**0.5, 
                                   end=beta_end**0.5, 
                                   steps=num_training_steps, 
                                   dtype=torch.float32)**0.5
        self.alphas = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(input=self.alphas, dim=0)
        self.ones = torch.tensor([1.0])

        self.num_traning_steps = num_training_steps
        self.generator = generator
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_time(self, num_inference_time:int = 50):
        """
        Chia quá trình khử nhiễu thành số lượng bước mong muốn (ví dụ: 50 thay vì 1000).
        Trong thực tế, mô hình không cần thiết phải khử nhiễu qua 1000 bước như khi huấn luyện. 
        Giảm số bước giúp suy luận nhanh hơn mà vẫn giữ được chất lượng ảnh.
        """
        self.num_inference_time = num_inference_time
        ratio_step = self.num_traning_steps // self.num_inference_time # 20
        timesteps = (np.arange(0, self.num_inference_time) * ratio_step).round()[::-1] # [1000, 980, 960...0]
        timesteps = timesteps.copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def set_strength(self, strength: float=0.7):
        """
        Xác định mức độ thay đổi của ảnh đầu vào khi tạo ảnh mới.
        - Nếu đặt strength = 1, ảnh đầu vào bị biến đổi mạnh hơn → kết quả đầu ra khác xa ảnh gốc.
        - Nếu đặt strength = 0, anh đầu ra gần giống ảnh gốc, quá trình khử nhiễu không có tác dụng rõ rệt.
        """
        start_step = self.num_inference_time - int(self.num_inference_time * strength)
        self.timesteps = self.timesteps[start_step:]
        # self.start_step = start_step

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Xác định thời điểm trước đó trong quá trình khử nhiễu.
        Trong khuếch tán ngược, mỗi bước sẽ dựa vào bước trước đó để cập nhật ảnh. 
        Do đó, biết bước thời gian trước giúp tính toán hệ số nhiễu chính xác.
        """
        time_prev = timestep - self.num_traning_steps // self.num_inference_time
        return time_prev

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Tính toán ảnh ở bước trước đó dựa trên ảnh hiện tại và đầu ra từ mô hình.
            1. Tính toán các hệ số nhiễu và ảnh gốc ước tính.
            2. Dự đoán ảnh ở bước trước đó dựa trên ảnh hiện tại.
            3. Thêm một phần nhiễu để đảm bảo quá trình suy luận không hoàn toàn xác định.
        Args:
            timstep: Thời điểm hiện tại trong quá trình reverese
            latents: Ảnh được encode sau khi đi qua VAE encoder
            model_ouput: latents đươc dự đoán từ Unet model

        Nếu không thêm nhiễu, kết quả sẽ quá "sạch" và có thể không phản ánh đúng phân phối ảnh thực tế.
        """
        time_curr = timestep
        time_prev = self._get_previous_timestep(timestep=time_curr)

        # 1. compute alphas, betas 
        alpha_prod_t = self.alpha_cumprod[time_curr]
        alpha_prod_prev_t = self.alpha_cumprod[time_prev] if time_prev >= 0 else self.ones

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_prev_t

        alpha_curr_t = alpha_prod_t / alpha_prod_prev_t
        beta_curr_t = 1 - alpha_curr_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - (beta_prod_t**0.5) * model_output) // alpha_prod_t ** 0.5

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (((alpha_prod_t - 1)**0.5)*beta_curr_t) / beta_prod_t
        current_sample_coeff = (alpha_curr_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # 4. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff*pred_original_sample + current_sample_coeff*model_output

        # 5. Add noise
        variance = 0
        if time_curr > 0:
            device = model_output.device
            noise = torch.randn(size=model_output.shape, generator=self.generator, 
                                device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            beta_t_variance = (beta_prod_t_prev / beta_prod_t ) * beta_curr_t
            beta_t_variance = torch.clamp(beta_t_variance, min=1e-20)
            variance = (beta_t_variance**0.5) * noise # [Bt_hat * I] 
        
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, 
                  original_samples: torch.FloatTensor,
                  timestep: torch.IntTensor) -> torch.FloatTensor:
        """
        Tạo ảnh nhiễu từ ảnh gốc để sử dụng trong quá trình huấn luyện mô hình.
            1. Lấy mẫu nhiễu ngẫu nhiên từ phân phối chuẩn.
            2. Áp dụng công thức từ paper DDPM để tạo ảnh nhiễu.
        """

        # tensor([1])
        sqrt_alpha_prod = self.alpha_cumprod[timestep] ** 0.5
        # tensor([1])
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        # [1] -> [1, 1, 1, 1]
        while(len(sqrt_alpha_prod) < len(original_samples)):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        # tensor([1])
        sqrt_one_minus_alpha_prod = (1 - self.alpha_cumprod[timestep]) ** 0.5
        # tensor([1])
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()

        # [1] -> [1, 1, 1, 1]
        while(len(sqrt_one_minus_alpha_prod) < len(original_samples)):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # from images/kernel_trick.png
        noise = torch.randn(size=original_samples.shape,
                            generator=self.generator,
                            device=original_samples.device,
                            dtype=original_samples.dtype)
        # [B, 4, latent_H, latent_W] -> [B, 4, latent_H, latent_W
        noise_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noise_samples # [B, 4, latent_H, latent_W]
    

def debug():
    sampler = DDPMSampler(
        generator=torch.Generator(device="cpu")
    )
    # ====================================
    # ------ test add noise method -------
    # ====================================
    # latents = torch.randn(size=(1, 4, 64, 64))
    # noise_added = sampler.add_noise(original_samples=latents,
    #                                 timestep=torch.tensor([1]))
    # print(noise_added.shape)

    # ====================================
    # ------ test step method ------------
    # ====================================
    timestep = 1
    latents = torch.randn(size=(1, 4, 64, 64))
    model_output = torch.randn(size=(1, 4, 64, 64))
    sampler.set_inference_time(num_inference_time=50)
    sampler.set_strength(strength=0.5)
    pred_prev_sample = sampler.step(timestep=timestep, latents=latents, model_output=model_output)
    print(pred_prev_sample.shape)

if __name__ == "__main__":
    debug()