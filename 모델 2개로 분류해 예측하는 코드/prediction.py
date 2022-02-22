# Evaluate
      if args.logging_steps > 0 and global_step % args.logging_steps == 0:
        if args.evaluate_test_during_training:
          evaluate(model, dev_dataset_first, mode="dev_first", _global_step=global_step) # 모델 1
        
        else:
          evaluate(model, test_dataset, mode="test", _global_step=global_step)
      # End Evaluate [if]

      # Save model checkpoint
      if args.save_steps > 0 and global_step % args.save_steps == 0:
        output_dir_first = os.path.join(args.output_dir_first, f"checkpoint-{global_step}")
        if not os.path.exists(output_dir_first):
          os.makedirs(output_dir_first)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir_first)

        torch.save(args, os.path.join(output_dir_first, "training_args.bin"))
        print(f" Saving model checkpoint to {output_dir_first}")

        if args.save_optimizer:
          torch.save(optimizer.state_dict(), os.path.join(output_dir_first, "optimizer.pt"))
          torch.save(scheduler.state_dict(), os.path.join(output_dir_first, "scheduler.pt"))
