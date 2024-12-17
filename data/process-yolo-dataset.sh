#!/usr/bin/env bash

ROOT_DIR="$1"
BASE_DIR="$2"

if [[ -z "$1" ]]; then
	printf "Usage: %s <root_dir> <base_dir>\n" "$0"
	exit 1
fi

create_directory() {
	if [[ -d "$BASE_DIR/$1" ]]; then
		printf "Directory already exists\n"
	else
		mkdir "$BASE_DIR/$1"
	fi
}

extract_image_name() {
	label_file="$1"
	image_file="${label_file%.txt}.jpg"
	image_file="${image_file/labels/images}"
	if [[ ! -f "$image_file" ]]; then
		printf "Image %s not found\n" "$image_file"
		exit 1
	fi
	echo "$image_file"
}

multiply() {
	a="$1"
	b="$2"
	printf "%.0f" "$(echo "$a * $b" | bc)"
}

subtract() {
	a="$1"
	b="$2"
	printf "%.0f" "$(echo "$a - $b" | bc)"
}

create_crop_str() {
	# Takes in width, height, x, y, w, h
	read -r width height x y w h <<<"$@"
	x_center="$(multiply "$x" "$width")"
	y_center="$(multiply "$y" "$height")"
	w="$(multiply "$w" "$width")"
	h="$(multiply "$h" "$height")"
	w_half="$(multiply "$w" "0.5")"
	h_half="$(multiply "$h" "0.5")"
	x="$(subtract "$x_center" "$w_half")"
	y="$(subtract "$y_center" "$h_half")"
	printf "%dx%d+%d+%d\n" "$w" "$h" "$x" "$y"
}

create_output_file() {
	fullname="$1"
	class="$2"
	filename="${fullname##*/}"
	printf "%s/%s/%s\n" "$BASE_DIR" "$class" "$filename"
}

for label_file in "$ROOT_DIR"/labels/*; do
	image_file=$(extract_image_name "$label_file")
	read -r width height <<<"$(identify -format "%w %h\n" "$image_file")"
	while read -r line; do
		wordcount=$(echo "$line" | wc -w)
		if [[ $wordcount -eq 5 ]]; then
			read -r class x y w h <<<"$line"
			create_directory "$class"
			crop_str=$(create_crop_str "$width" "$height" "$x" "$y" "$w" "$h")
			output_file=$(create_output_file "$image_file" "$class")
			convert "$image_file" -crop "$crop_str" "$output_file" &&
				printf "Created %s\n" "$output_file"
		fi
	done <"$label_file"
done
