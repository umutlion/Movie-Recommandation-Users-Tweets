<?php

namespace Database\Seeders;

use App\Models\Movie;
use Illuminate\Database\Seeder;
use Illuminate\Support\Facades\DB;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     *
     * @return void
     */
    public function run()
    {
        $first = DB::table('mytable')->get();
        foreach ($first as $fr) {
            $film = new Movie;
            $film->overview = $fr->overview;
            $film->imdb_id = str_replace("tt","",str_replace("tt0","",$fr->imdb_id));
            $film->title = $fr->title;
            $film->results = (float) str_replace(["[[", "]]"], ["", ""], $fr->results);
            if ($film->results) {
                $film->save();
            }
        }
    }
}
