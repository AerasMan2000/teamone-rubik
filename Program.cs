using System;
using System.IO;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ImageProcessing
{
    class Program
    {
        static void Main(string[] args)
        {
            const float width = 128, height = 128;
            int counter = 0;
            int index = 0;

            // Loops through all the existing images for processing
            foreach (var file in Directory.GetFiles(@"D:\Project Files\ImageProcessing\ImageProcessing\imgs"))
            {
                var image = Image.FromFile(file);

                if (!(image.Width >= width && image.Height >= height))
                {
                    counter++;
                    continue;
                }

                var mult = Math.Max(image.Width / width, image.Height / height);

                int lWidth = (int)Math.Min(image.Width / mult, width), 
                    lHeight = (int)Math.Min(image.Height / mult, height);

                var destRect = new Rectangle(0, 0, lWidth, lHeight);
                var destImage = new Bitmap(lWidth, lHeight);
    
                destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

                // Draws the original to the smaller size.
                using (var graphics = Graphics.FromImage(destImage))
                {
                    graphics.CompositingMode = CompositingMode.SourceCopy;
                    graphics.CompositingQuality = CompositingQuality.HighQuality;
                    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                    using (var wrapMode = new ImageAttributes())
                    {
                        wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                        graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                    }
                }

                // Saving and clean up
                try
                {
                    destImage.Save(@"D:\Project Files\ImageProcessing\ImageProcessing\out\" + $"{index}.png", ImageFormat.Png);
                }
                catch (ExternalException e)
                {
                    Console.WriteLine(e.StackTrace);
                }

                index++;
                image.Dispose();
                destImage.Dispose();
            }
            Console.WriteLine(counter);
            Console.WriteLine(index);
        }
    }
}
