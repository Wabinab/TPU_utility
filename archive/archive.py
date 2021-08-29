# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-25T05:58:03.475507Z","iopub.execute_input":"2021-08-25T05:58:03.475876Z","iopub.status.idle":"2021-08-25T05:58:03.619021Z","shell.execute_reply.started":"2021-08-25T05:58:03.475841Z","shell.execute_reply":"2021-08-25T05:58:03.617924Z"}}
# class LRFinder:
    
#     def __init__(self, model, opt, criterion, train_loader=None, val_loader=None,
#                  start_lr=1e-7, device=None):
#         self.model = copy.deepcopy(model)
#         self.opt = opt
#         self.criterion = criterion
#         self.train_loader = train_loader
#         self.val_loader = val_loader
        
#         # Save model initial dict
# #         self.save_file = Path("./")
# #         torch.save(self.model, self.save_file)
        
#         if device is None: self.device = next(model.parameters()).device
#         else: self.device = device
            
#         self.model.to(self.device)
#         if self.device.type == "xla": print("Using TPU")  # as an assertion
        
#         self.history = {"lr": [], "losses": []}
        
#         for l in self.opt.param_groups:
#             l["initial_lr"] = start_lr
            
#     def reset(self):
#         """Resets model to initial state."""
# #         self.model = torch.load(self.save_file)
# #         self.model.train()
# #         self.save_file.unlink()
# #         return self.model
#         return None
    
#     # def calculate_smoothing_value(self, beta):
#     #     n, mov_avg = 0, 0
#     #     while True:
#     #         n += 1
#     #         value = yield
#     #         mov_avg = beta * mov_avg + (1 - beta) * value
#     #         smooth = mov_avg / (1 - beta**n)
#     #         yield smooth

#     def calculate_smoothing_value(self, beta):
#         while True:
#             loss, prev_loss = yield
#             loss = beta * loss + (1 - beta) * prev_loss
#             yield loss

            
#     def lr_find(self, train_loader=None, val_loader=None, end_lr=10, num_iter=150, step_mode="exp", 
#                loss_smoothing_beta=0.05, diverge_th=5, non_blocking=True):
#         """
#         Performs LR Find test
        
#         :args:
#             train_loader: data loader. 
#             end_lr: maximum lr to stop. 
#             num_iter: max iterations.
#             step_mode: anneal function. Default 'exp'. Choices 'linear', 'cos'. 
#             loss_smoothing_beta: loss smoothing factor. Range: [0, 1). Defaults: 0.05.
#             diverge_th: max loss value after which training should be stopped. 
#             non_blocking: (bool) Whether to have non-blocking transfer between device. 
#         """
#         # Reset test results
#         self.history = {"lr": [], "losses": []}
#         self.best_loss = None
#         self.smoothener = self.calculate_smoothing_value(loss_smoothing_beta)
        
#         choices = {
#             "exp": ExponentialScheduler,
#             "cos": CosineScheduler,
#             "linear": LinearScheduler
#         }
        
#         try: lr_scheduler = choices[step_mode.lower()](self.opt, end_lr, num_iter)
#         except KeyError: 
#             raise ValueError(f"Expected mode 'exp', 'cos', or 'linear'; got {step_mode}")
            
#         if 0 < loss_smoothing_beta >= 1:
#             raise ValueError("loss_smoothing_beta outside range [0, 1).")
            
#         if train_loader is None: train_loader = self.train_loader
#         assert train_loader is not None
#         iterator = iter(train_loader)

#         if val_loader is None: val_loader = self.val_loader
#         val_iter = iter(val_loader) if val_loader is not None else None

        
#         for each_iter in tqdm(range(num_iter)):
#             try: data, target = next(iterator)
#             except StopIteration: 
#                 iterator = iter(train_loader)
#                 data, target = next(iterator)
                
#             loss = self._train_batch(data, target.to(torch.float32), non_blocking=non_blocking)
#             if val_loader: val_loss = self._validate(val_iter, non_blocking=non_blocking)
            
#             # Update learning rate
#             self.history["lr"].append(lr_scheduler.get_lr()[0])
#             lr_scheduler.step()
            
#             # Track best loss and smooth if loss_smoothing_beta is specified.
#             if each_iter == 0: self.best_loss = loss
#             else:
#                 next(self.smoothener)
#                 # self.best_loss = self.smoothener.send(loss)
#                 self.best_loss = self.smoothener.send((loss, self.history["losses"][-1]))
#                 if loss < self.best_loss: self.best_loss = loss
                    
#             # Check if loss diverged. If it does, stop the test.
#             self.history["losses"].append(loss)
#             if loss > diverge_th * self.best_loss: break
                
                
#         clear_output()
#         print(f"Best train loss: {self.best_loss}")
#         # print(f"Last val loss: {val_loss}")
#         steepest, index = self.steepest_lr()
#         print(f"Steepest point: {steepest}")
#         self._plot(index=index)
#         _ = self.reset()  # used if model is saved. One haven't make this work, yet. 
        
#         return steepest, _, self.best_loss
        
#     def _train_batch(self, data, target, non_blocking=True):
#         self.model.train()  # training mode
#         data = data.to(self.device, non_blocking=non_blocking)
#         target = target.to(self.device, non_blocking=non_blocking)
        
#         # Forward pass
#         self.opt.zero_grad()
#         output = self.model(data)
#         loss = self.criterion(output, target)
        
#         # Backward pass
#         loss.backward()
#         if self.device.type == "xla": xm.optimizer_step(self.opt, barrier=True)
#         else: self.opt.step()
        
#         return loss.item()
    
#     def _plot(self, index=None):
#         """
#         index: index where steepest value lies. 
#         """
#         losses = self.history["losses"]
#         lr = self.history["lr"]
        
#         plt.figure(dpi=120)
#         plt.plot(lr, losses)
#         # plt.semilogx(lr, losses)
#         if index: plt.scatter(lr[index], losses[index], s=75, marker="o", color="red", zorder=3)
#         plt.xscale("log")
#         plt.xlabel("Learning Rate")
#         plt.ylabel("Losses")
#         plt.grid()
#         plt.show()

#     def _validate(self, val_iter, non_blocking=True):
#         running_loss = 0
#         self.model.eval()
        
#         with torch.no_grad():
#             for data, target in val_iter:
#                 data = data.to(self.device, non_blocking=True)
#                 target = target.to(self.device, non_blocking=True)

#                 output = self.model(data)
#                 loss = self.criterion(output, target)
#                 running_loss += loss.item() * len(target)

#         return running_loss / len(val_iter.dataset)
    
#     def steepest_lr(self, skip_end=5):
#         losses = np.array(self.history["losses"])
#         lr = np.array(self.history["lr"])
#         if skip_end != 0: losses, lr = losses[:-skip_end], lr[:-skip_end]

#         # Suggest learning rate:
#         index = (np.gradient(losses)).argmin()
#         return lr[index], index
#         # return lr[np.argmax(losses[:-1] - losses[1:])]