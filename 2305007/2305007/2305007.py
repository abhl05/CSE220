import numpy as np

class DiscreteSignal :
    
    def __init__(self, start_time, end_time) :
        self.start_time = start_time
        self.end_time = end_time 
        self.values = np.zeros(end_time - start_time + 1)

    def set_value_at_time(self, t, value) :
        if self.start_time <= t <= self.end_time :
            self.values[t - self.start_time] = value
        else :
            raise IndexError
        
    def get_value_at_time(self, t) :
        if self.start_time <= t <= self.end_time :
            return self.values[t - self.start_time] 
        return 0.0
    
    def shift(self, k) :
        new_start = self.start_time + k
        new_end = self.end_time + k

        shifted_signal = DiscreteSignal(new_start, new_end)
        
        shifted_signal.values = np.copy(self.values)
        return shifted_signal
    
    def display(self):
        indices = np.arange(self.start_time, self.end_time)
        print(f"n {indices.tolist()}")
        print(f"x[n] {self.values.tolist()}")
    

class LTISystem:
    def __init__(self, impluse_response: DiscreteSignal) :
        self.h = impluse_response
    
    def output(self, x: DiscreteSignal) -> DiscreteSignal:
        y_start = x.start_time + self.h.start_time
        y_end = x.end_time + self.h.end_time

        y = DiscreteSignal(y_start, y_end)

        # y[n] = sum(x[n]*h[n-k])

        k_array = np.arange(x.start_time, x.end_time + 1)
        x_array = x.values

        for n in range(y_start, y_end) :
            h_indices = n - k_array
            
            # h_indices গুলোর মধ্যে যেগুলো h এর সীমানার ভেতরে আছে তা বের করার মাস্ক
            valid_mask = (h_indices >= self.h.start_time) & (h_indices <= self.h.end_time)
            
            # কনভোলিউশন সাম সূত্র: sum( x[k] * h[n - k] )
            # শুধুমাত্র ভ্যালিড ইনডেক্সগুলোর মান নিয়ে গুণ করা হচ্ছে, বাকিগুলো ০
            current_y_val = 0.0
            for idx, k_val in enumerate(k_array):
                if valid_mask[idx]:
                    h_val = self.h.get_value_at_time(h_indices[idx])
                    current_y_val += x_array[idx] * h_val
            
            y.set_value_at_time(n, current_y_val)
            
        return y
    
if __name__ == "__main__":
    # ইনপুট সিগনাল x[n] তৈরি (ধরি n = 0, 1, 2 এর জন্য মান 1, 2, 3)
    x = DiscreteSignal(0, 2)
    x.set_value_at_time(0, 1)
    x.set_value_at_time(1, 2)
    x.set_value_at_time(2, 3)
    
    # ইমপালস রেসপন্স h[n] তৈরি (ধরি n = 0, 1 এর জন্য মান 1, -1)
    h = DiscreteSignal(0, 1)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, -1)
    
    # LTI সিস্টেম তৈরি ও আউটপুট গণনা
    system = LTISystem(h)
    y = system.output(x)
    
    print("Input Signal x[n]:")
    x.display()
    print("\nImpulse Response h[n]:")
    h.display()
    print("\nOutput Signal y[n] (Convolution Result):")
    y.display()


import numpy as np
import matplotlib.pyplot as plt

# [পূর্বের DiscreteSignal এবং LTISystem ক্লাস দুটি এখানে যুক্ত থাকবে]

# ১. স্যাম্পল ডাটা তৈরি
x = DiscreteSignal(0, 4)
for i, val in enumerate([1, 2, 3, 2, 1]):
    x.set_value_at_time(i, val)

h = DiscreteSignal(-1, 1)
h.set_value_at_time(-1, 0.5)
h.set_value_at_time(0, 1.0)
h.set_value_at_time(1, 0.5)

# ২. আউটপুট গণনা
system = LTISystem(h)
y = system.output(x)

# ৩. ম্যাটপ্লটলিব দিয়ে প্লট তৈরি
plt.figure(figsize=(12, 8))

# ইনপুট প্লট
plt.subplot(3, 1, 1)
plt.stem(np.arange(x.start_time, x.end_time + 1), x.values, 'b', markerfmt='bo', basefmt='k-')
plt.title("Input Signal $x[n]$")
plt.grid(True)

# ইমপালস রেসপন্স প্লট
plt.subplot(3, 1, 2)
plt.stem(np.arange(h.start_time, h.end_time + 1), h.values, 'g', markerfmt='go', basefmt='k-')
plt.title("Impulse Response $h[n]$ (Moving Average)")
plt.grid(True)

# আউটপুট প্লট
plt.subplot(3, 1, 3)
plt.stem(np.arange(y.start_time, y.end_time + 1), y.values, 'r', markerfmt='ro', basefmt='k-')
plt.title("Output Signal $y[n]$ (Convolution Result)")
plt.grid(True)

plt.tight_layout()
plt.show()


import cv2  # ছবি লোড করার জন্য (pip install opencv-python)

# একটি স্যাম্পল গ্রেস্কেল ইমেজ লোড করুন বা ডামি ইমেজ তৈরি করুন
# আমরা পরীক্ষার জন্য একটি ১০০x১০০ ডামি ইমেজ বানিয়ে নিচ্ছি যার মাঝে একটি সাদা স্কয়ার আছে
image = np.zeros((100, 100))
image[30:70, 30:70] = 255  # মাঝখানে একটি শার্প সাদা বক্স

# ব্লার ফিল্টার (Box Blur Kernel) তৈরি
# ১D সিগনাল হিসেবে ফিল্টারের উইন্ডো সাইজ ৫ নিলাম, যার সব মান সমান (সবগুলোর যোগফল ১ হতে হবে)
kernel_size = 5
blur_kernel = DiscreteSignal(-(kernel_size//2), kernel_size//2)
for t in range(blur_kernel.start_time, blur_kernel.end_time + 1):
    blur_kernel.set_value_at_time(t, 1.0 / kernel_size)

# LTI সিস্টেম ইনিশিয়েট করা
blur_system = LTISystem(blur_kernel)

# নতুন ব্লার ইমেজ রাখার জন্য খালি ম্যাট্রিক্স
blurred_image = np.zeros_like(image)

# ছবির প্রতিটি রো (Row) এর ওপর ১D কনভোলিউশন চালানো
for row_idx in range(image.shape[0]):
    # ১. কারেন্ট রো-কে DiscreteSignal এ রূপান্তর
    row_signal = DiscreteSignal(0, image.shape[1] - 1)
    row_signal.values = np.copy(image[row_idx, :])
    
    # ২. LTI সিস্টেম দিয়ে কনভোলিউশন করা
    output_signal = blur_system.output(row_signal)
    
    # ৩. আউটপুট সিগনাল থেকে শুধুমাত্র ছবির অরিজিনাল সাইজের (০ থেকে width-1) অংশটুকু কেটে নেওয়া
    # কারণ কনভোলিউশনের ফলে সিগনালের সাইজ কিছুটা বেড়ে যায় (Padding effect)
    for col_idx in range(image.shape[1]):
        blurred_image[row_idx, col_idx] = output_signal.get_value_at_time(col_idx)

# ছবি দুটি পাশাপাশি রেখে ভিজ্যুয়ালাইজ করা
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Sharp Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title("1D Blurred Image (Horizontal)")
plt.axis('off')

plt.tight_layout()
plt.show()
